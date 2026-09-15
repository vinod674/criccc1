import hashlib
import logging
import os
import re
import sqlite3
import time
import urllib.parse
import traceback
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup
from groq import Groq

# =====================
# CONFIGURATION
# =====================
BOT_TOKEN    = os.getenv("BOT_TOKEN")
CHAT_ID      = os.getenv("CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
IST     = timezone(timedelta(hours=5, minutes=30))

RESULT_PHRASES = [
    "won by", "win by", "match drawn", "match tied",
    "abandoned", "no result", "beat", "beats",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_ist_now():
    return datetime.now(IST)


# =====================
# DATABASE SETUP
# =====================
try:
    conn   = sqlite3.connect("cricket_final.db", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY)")
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS state "
        "(m_id TEXT PRIMARY KEY, last_over REAL, last_wickets INTEGER, "
        "toss_done INTEGER DEFAULT 0, innings INTEGER DEFAULT 1)"
    )
    cursor.execute("CREATE TABLE IF NOT EXISTS daily_logs (date TEXT PRIMARY KEY)")
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS tracking_config "
        "(m_id TEXT PRIMARY KEY, match_name TEXT, is_active INTEGER DEFAULT 1)"
    )

    # Safe column additions — silently ignored if the column already exists
    for _col_sql in [
        "ALTER TABLE state ADD COLUMN last_wicket_over REAL DEFAULT -10.0",
        "ALTER TABLE state ADD COLUMN innings INTEGER DEFAULT 1",
        "ALTER TABLE state ADD COLUMN last_double_strike_wk INTEGER DEFAULT 0",
        "ALTER TABLE state ADD COLUMN last_score INTEGER DEFAULT 0",
    ]:
        try:
            cursor.execute(_col_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
except Exception as e:
    logger.error("Database Initialization Error: %s", e)

# In-memory session cache (complements DB; resets on bot restart)
match_state    = {}
last_update_id = None


# =====================
# TEAM ABBREVIATION HELPERS
# =====================
_COUNTRY_ABBR = {
    "India":         "IND",  "Australia":    "AUS",  "England":      "ENG",
    "Pakistan":      "PAK",  "New Zealand":  "NZ",   "South Africa": "SA",
    "Sri Lanka":     "SL",   "West Indies":  "WI",   "Bangladesh":   "BAN",
    "Zimbabwe":      "ZIM",  "Afghanistan":  "AFG",  "Ireland":      "IRE",
    "Scotland":      "SCO",  "Netherlands":  "NED",  "Namibia":      "NAM",
    "United States": "USA",  "Canada":       "CAN",  "Kenya":        "KEN",
    "Nepal":         "NEP",  "Oman":         "OMA",  "UAE":          "UAE",
}


def _abbr(team_name):
    """Return a short abbreviation, e.g. 'India' → 'IND', 'New Zealand' → 'NZ'."""
    if not team_name:
        return "???"
    return _COUNTRY_ABBR.get(team_name, team_name[:3].upper())


def _vs_tag(team_a, team_b):
    """Return a concise 'IND vs AUS' heading tag from full team names."""
    return f"{_abbr(team_a)} vs {_abbr(team_b)}"


# =====================
# AI ENGINE (GROQ)
# =====================
_TONE_MAP = {
    "IMPACT PLAYER":     "Analyse the strategic impact of this substitution — offensive chase move, or defensive to save wickets?",
    "STRATEGIC TIMEOUT": "Analyse the current run rate and momentum. Who is winning this phase?",
    "BIG OVER":          "Hype up this massive over! Emphasise the huge momentum shift and how the bowler is getting taken apart.",
    "POWERPLAY":         "Summarise the first 6 overs. Did the batting team dominate, or did the bowlers keep it tight?",
    "MATCH_END":         "Write a short, thrilling summary celebrating the winner. Highlight the margin of victory.",
}


def get_pro_edit(match_facts):
    """
    Call the Groq LLM to generate a polished, channel-ready WhatsApp post.
    Returns the AI text on success, or None on any failure (after retrying once).
    """
    if not groq_client or not match_facts:
        return None

    event_type      = match_facts.get("event_type", "")
    current_wickets = match_facts.get("wickets", 0)

    # Pick a tone instruction from the map; double-wicket varies by collapse depth
    custom_instruction = next(
        (instr for key, instr in _TONE_MAP.items() if key in event_type), None
    )
    if "DOUBLE_WICKET" in event_type:
        if current_wickets >= 5:
            custom_instruction = (
                "The team has lost two more wickets and the lower order is now exposed. "
                "Emphasise the continuing, disastrous collapse."
            )
        else:
            custom_instruction = (
                "Two quick wickets have fallen! Emphasise the sudden shock and the "
                "momentum shift towards the bowling side."
            )

    prompt = f"""You are a sharp cricket news editor for a fast-moving WhatsApp channel.
Turn the raw match data below into a SHORT, PUNCHY post. Readers are on their phones — get to the point fast.

MATCH THE STYLE OF THESE EXAMPLES EXACTLY:

EXAMPLE 1 (Toss):
🏏 TOSS UPDATE – ENG vs SL 🏏
Sri Lanka win the toss, elect to bowl first at Pallekele.

Early moisture on offer — the Lankan spinners will fancy this. Game on!

EXAMPLE 2 (Match Update):
🏏 10 OVER UPDATE – ENG vs SL 🏏
England 68/4 after 10, and it's scrappy.

Phil Salt (37*) is fighting alone as Sri Lanka's spinners choke the innings. The middle order needs to fire, fast.

---
STRICT CURRENT FACTS TO USE:
- Match: {match_facts.get('match_name', 'Unknown')}
- Event: {event_type}
- Batting Team (currently on strike): {match_facts.get('team_batting', 'Unknown')}
- Bowling Team: {match_facts.get('team_bowling', 'Unknown')}
- Current Innings: {match_facts.get('innings', 1)}
- Score: {match_facts.get('score_display', 'Unknown')}
- Official Status / Commentary: {match_facts.get('status_text', '')}

RULES — FOLLOW EXACTLY:
1. Exactly 1 heading, then 2 SHORT paragraphs.
2. Blank line between heading and each paragraph.
3. Each paragraph: 1-2 sentences, max ~30 words. Total across both paragraphs: max ~60 words.
4. No fluff, no filler adjectives. Every word should earn its place.
5. TONE: {custom_instruction if custom_instruction else "Keep it sharp, factual, and easy to skim in 2 seconds."}
6. STRICT: If 'Current Innings' is 2, DO NOT mention who won the toss. Focus only on the chase.
7. NEVER invent stats not given in 'STRICT CURRENT FACTS' above.
8. Do not use hashtags, emojis beyond the heading, or sign-offs.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an elite cricket news editor who writes tight, scannable "
                "WhatsApp updates. You never pad your writing — short and sharp always "
                "beats long and flowery. You mirror the user's style examples exactly."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for attempt in range(2):
        try:
            res = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0.6,
                max_tokens=300,   # room for a 2-paragraph short update
                top_p=0.9,
                reasoning_effort="low",  # minimize reasoning tokens, favor final answer
            )
            msg_obj = res.choices[0].message
            output  = (msg_obj.content or "").strip()
            if not output:
                logger.warning("Empty content. finish_reason=%s full_message=%s",
                                res.choices[0].finish_reason, msg_obj)
            logger.info("Groq SUCCESS - got %d chars: %s", len(output), output[:80])
            return output.replace("\n\n\n", "\n\n") if output else None
        except Exception as e:
            logger.warning("Groq API error (attempt %d): %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(1)
    return None


# =====================
# CORE UTILITIES
# =====================
def send_telegram(raw_text, pro_edit=False, match_facts=None):
    """
    Send a Telegram message to the configured channel.
    Always sends the raw message first (acts as fallback if AI glitches),
    then follows up with the AI-polished version if pro_edit=True.
    """
    if not raw_text or not BOT_TOKEN or not CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # Always send raw first
    try:
        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": raw_text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": "true",
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.warning("send_telegram raw failed: %s", exc)

    # Then follow up with AI version if available
    if pro_edit and GROQ_API_KEY and match_facts:
        ai_text = get_pro_edit(match_facts)
        logger.info("AI text ready to send: %s", bool(ai_text))
        if ai_text:
            try:
                res2 = requests.post(
                    url,
                    data={
                        "chat_id": CHAT_ID,
                        "text": ai_text,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": "true",
                    },
                    timeout=10,
                )
                logger.info("Telegram AI send status: %s | %s", res2.status_code, res2.text[:200])
            except requests.RequestException as exc:
                logger.warning("send_telegram AI failed: %s", exc)


def get_img_link(query):
    """Return a Google Images search URL for the given cricket query."""
    safe_query = urllib.parse.quote(f"{query} Cricket Match {get_ist_now().year}")
    return f"https://www.google.com/search?q={safe_query}&tbm=isch"


def overs_to_balls(overs):
    """Convert decimal overs '14.3' (or float 14.3) to total balls (87)."""
    if not overs and overs != 0:
        return 0
    m = re.match(r"^(\d+)(?:\.(\d))?$", str(overs).strip())
    if not m:
        return 0
    whole = int(m.group(1))
    balls = min(max(int(m.group(2) or 0), 0), 5)
    return whole * 6 + balls


def parse_overs(raw):
    """
    Robustly extract a float over count from raw Cricbuzz strings.
    Handles '14.3', '(14.3)', '14.3 Ov', and similar variants.
    Falls back to 0.0 if no numeric pattern is found.
    """
    if not raw:
        return 0.0
    m = re.search(r"(\d+(?:\.\d)?)", str(raw).strip())
    return float(m.group(1)) if m else 0.0


def stable_event_suffix(text):
    """10-char SHA-1 hash of text — used to build stable dedup event IDs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# =====================
# MATCH FILTERING
# =====================
def is_international_text_check(text):
    """
    Returns True if the match should be tracked.
    Accepts: IPL, WPL, and international bilateral/ICC events.
    Rejects: junior cricket, A-team matches, domestic leagues.
    """
    title = text.upper()

    if any(phrase in title for phrase in ["INDIAN PREMIER LEAGUE", " IPL ", "TATA IPL", "WPL"]):
        return True

    ipl_teams = [
        "MUMBAI INDIANS", "CHENNAI SUPER KINGS", "ROYAL CHALLENGERS BENGALURU",
        "ROYAL CHALLENGERS BANGALORE", "KOLKATA KNIGHT RIDERS", "SUNRISERS HYDERABAD",
        "RAJASTHAN ROYALS", "DELHI CAPITALS", "PUNJAB KINGS",
        "LUCKNOW SUPER GIANTS", "GUJARAT TITANS",
    ]
    if any(team in title for team in ipl_teams):
        return True

    ipl_abbrevs = [
        r"\bRCB\b", r"\bCSK\b", r"\bMI\b", r"\bKKR\b", r"\bSRH\b",
        r"\bRR\b",  r"\bDC\b",  r"\bPBKS\b", r"\bLSG\b", r"\bGT\b",
    ]
    if any(re.search(abbrev, title) for abbrev in ipl_abbrevs):
        return True

    # Exclude junior / A-team / truly domestic events.
    # NOTE: "TROPHY" intentionally NOT in this list — ICC Champions Trophy /
    # World Cup titles also contain it. Domestic trophies won't have 2 country
    # names in the title anyway, so the countries check below handles them.
    if any(x in title for x in [
        " U19", " U-19", "LEAGUE", " XI", "INDIA A", "PAKISTAN A",
        "AUSTRALIA A", "SRI LANKA A", "ENGLAND LIONS", "HONG KONG", "CHINA",
    ]):
        return False

    countries = [
        "INDIA", "AUSTRALIA", "ENGLAND", "NEW ZEALAND", "SOUTH AFRICA",
        "PAKISTAN", "SRI LANKA", "WEST INDIES", "BANGLADESH", "ZIMBABWE",
        "AFGHANISTAN", "IRELAND", "SCOTLAND", "NETHERLANDS", "NAMIBIA",
        "UNITED STATES", "CANADA", "KENYA", "NEPAL", "OMAN", "UAE",
    ]
    return sum(1 for c in countries if c in title) >= 2


def is_result_text(text):
    """Return True if the status string indicates a completed match."""
    lower = (text or "").lower()
    return any(phrase in lower for phrase in RESULT_PHRASES)


def is_womens_match(match_name):
    """Women's matches are muted by default; user can enable via /track."""
    name_up = match_name.upper()
    return "WOMEN" in name_up or " W " in name_up or name_up.endswith(" W") or "WPL" in name_up


def get_teams_from_name(match_name):
    """
    Parse 'Team A vs Team B, Series Info' → clean (team_a, team_b) strings.
    Strips trailing series/match labels like '3rd T20I', '1st ODI', 'Test'.
    """
    teams = [t.strip() for t in re.split(r'\s+vs\s+|\s+v\s+', match_name, flags=re.IGNORECASE)]
    if len(teams) >= 2:
        def _clean(t):
            t = re.sub(
                r'[,]?\s*\d*\s*(?:\d+(?:st|nd|rd|th)\s+)?(?:T20I|T20|ODI|Test|OD)\b.*$',
                '', t, flags=re.IGNORECASE,
            )
            t = re.sub(r',.*$', '', t)
            return t.strip()
        team_a = _clean(teams[0]) or teams[0]
        team_b = _clean(teams[1]) or teams[1]
        return team_a, team_b
    return "Team A", "Team B"


# =====================
# SCRAPING ENGINE HELPERS
# =====================
def scrape_todays_schedule():
    """Fetch today's international match schedule from Cricbuzz and format it."""
    try:
        response       = requests.get("https://www.cricbuzz.com/cricket-schedule", headers=HEADERS, timeout=15)
        soup           = BeautifulSoup(response.text, "html.parser")
        today_str      = get_ist_now().strftime("%a %b %d").upper()
        todays_matches = []

        for block in soup.find_all("div", class_="cb-col-100 cb-col cb-schdl"):
            date_header = block.find("div", class_="cb-col-100 cb-col cb-lv-grn-strip")
            if not date_header or today_str not in date_header.get_text().upper():
                continue
            match_list = block.find_next_sibling("div")
            if not match_list:
                continue
            for match in match_list.find_all("div", class_="cb-ovr-flo"):
                match_info = match.get_text(strip=True)
                if is_international_text_check(match_info):
                    todays_matches.append(f"• {match_info}")

        if not todays_matches:
            return "No major matches scheduled for today."

        header = (
            f"📅 *TODAY'S CRICKET SCHEDULE*\n"
            f"—————————————————\n"
            f"_{get_ist_now().strftime('%d %B %Y')}_\n\n"
        )
        footer = (
            f"\n\n🖼 [Tap for Series Graphics]({get_img_link('Cricket Schedule')})\n"
            f"—————————————————\n"
            f"🔔 *Keep notifications ON for live updates!*"
        )
        return header + "\n".join(todays_matches) + footer
    except Exception as exc:
        logger.warning("Schedule scrape failed: %s", exc)
        return None


def handle_daily_briefing():
    """Send today's schedule once per day at or after 08:00 IST."""
    now        = get_ist_now()
    today_date = now.strftime("%Y-%m-%d")
    if now.hour >= 8:
        row = cursor.execute("SELECT date FROM daily_logs WHERE date=?", (today_date,)).fetchone()
        if not row:
            brief = scrape_todays_schedule()
            # Only send if there are actual matches — don't spam "no matches today"
            if brief and brief != "No major matches scheduled for today.":
                send_telegram(brief)
            # Always mark as done so we don't retry every 15s all day
            cursor.execute("INSERT INTO daily_logs (date) VALUES (?)", (today_date,))
            conn.commit()


def _command_matches(text, command):
    return text.strip().startswith(command)


# =====================
# COMMAND HANDLER
# =====================
def handle_commands():
    """Poll Telegram getUpdates for bot commands: /tracklist /track /stop /score."""
    global last_update_id
    if not BOT_TOKEN:
        return

    url    = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {"timeout": 5}
    if last_update_id is not None:
        params["offset"] = last_update_id + 1

    try:
        res = requests.get(url, params=params, timeout=10).json()
        if not res.get("ok"):
            return

        for update in res.get("result", []):
            last_update_id = update["update_id"]
            msg_data = update.get("message") or update.get("channel_post")
            if not msg_data:
                continue
            text = msg_data.get("text", "")

            if _command_matches(text, "/help"):
                send_telegram(
                    "🏏 *CRICKET BOT – COMMANDS*\n"
                    "—————————————————\n"
                    "*/tracklist* — Show all live matches & tracking status\n"
                    "*/track <n>* — Start tracking match number n\n"
                    "*/stop <n>* — Mute match number n\n"
                    "*/score* — Get current live scores\n"
                    "*/help* — Show this message\n"
                    "—————————————————\n"
                    "_Women's matches are muted by default. Use /track to enable._"
                )

            elif _command_matches(text, "/tracklist"):
                matches = scrape_match_links()
                if not matches:
                    send_telegram("📭 No LIVE matches found right now.")
                else:
                    report = "📋 *TRACKING MANAGER*\n—————————————————\n"
                    for i, (name, link) in enumerate(matches):
                        m_id           = link.split("/")[-2]
                        row            = cursor.execute(
                            "SELECT is_active FROM tracking_config WHERE m_id=?", (m_id,)
                        ).fetchone()
                        default_active = 0 if is_womens_match(name) else 1
                        is_active      = row[0] if row else default_active
                        status         = "✅ Tracking" if is_active == 1 else "❌ Muted"
                        report += (
                            f"*{i + 1}.* {name}\n"
                            f"Status: {status}\n"
                            f"Toggle: `/track {i + 1}` or `/stop {i + 1}`\n\n"
                        )
                    send_telegram(report)

            elif _command_matches(text, "/track"):
                try:
                    idx     = int(text.split()[-1]) - 1
                    matches = scrape_match_links()
                    name, link = matches[idx]
                    m_id = link.split("/")[-2]
                    cursor.execute("INSERT OR REPLACE INTO tracking_config VALUES (?, ?, 1)", (m_id, name))
                    conn.commit()
                    send_telegram(f"✅ Now tracking: *{name}*")
                except (ValueError, IndexError):
                    send_telegram("⚠️ Invalid ID. Use /tracklist to see active match numbers.")

            elif _command_matches(text, "/stop"):
                try:
                    idx     = int(text.split()[-1]) - 1
                    matches = scrape_match_links()
                    name, link = matches[idx]
                    m_id = link.split("/")[-2]
                    cursor.execute("INSERT OR REPLACE INTO tracking_config VALUES (?, ?, 0)", (m_id, name))
                    conn.commit()
                    send_telegram(f"❌ Successfully Muted: *{name}*")
                except (ValueError, IndexError):
                    send_telegram("⚠️ Invalid ID. Use /tracklist to see active match numbers.")

            elif _command_matches(text, "/score"):
                send_telegram("🏏 *Fetching live matches...*")
                matches = scrape_match_links()
                if not matches:
                    send_telegram("⚠️ There are no relevant matches on the board right now.")
                else:
                    summary_data = []
                    for name, link in matches[:5]:
                        score = scrape_instant_score(link)
                        summary_data.append(f"🔹 *{name}*\n{score}")
                    send_telegram(
                        "🏆 *LIVE MATCHES* 🏆\n—————————————————\n"
                        + "\n\n".join(summary_data)
                    )
    except Exception as e:
        logger.warning("Command Error: %s", e)


# =====================
# SCRAPING ENGINE
# =====================
def scrape_match_links():
    """Return list of (match_name, url) for all live international matches on Cricbuzz."""
    try:
        res  = requests.get(
            "https://www.cricbuzz.com/cricket-match/live-scores",
            headers=HEADERS, timeout=15,
        )
        soup    = BeautifulSoup(res.text, "html.parser")
        matches = []
        seen    = set()  # O(1) dedup instead of O(n²) list scan

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/live-cricket-scores/" not in href and "/cricket-scores/" not in href:
                continue
            name = a_tag.get("title", "").strip() or a_tag.get_text(separator=" ", strip=True)
            if not name or not is_international_text_check(name):
                continue
            full_link = ("https://www.cricbuzz.com" + href) if href.startswith("/") else href
            if full_link not in seen:
                seen.add(full_link)
                matches.append((name, full_link))
        return matches
    except Exception as e:
        logger.warning("match links scrape failed: %s", e)
        return []


def scrape_instant_score(match_url):
    """Return a one-line score string used by the /score command."""
    try:
        response  = requests.get(match_url, headers=HEADERS, timeout=15)
        soup      = BeautifulSoup(response.text, "html.parser")
        score_div = soup.find(
            "div",
            class_=lambda x: x and (("text-3xl" in x and "font-bold" in x) or "cb-font-20" in x),
        )
        if not score_div:
            return "Score not available yet"

        p = score_div.find_all("div")
        if not p:
            return "Score structure unavailable"

        runs    = p[0].get_text(strip=True)
        wickets = p[1].get_text(strip=True).replace("-", "") if len(p) > 1 else "0"
        overs   = (
            p[2].get_text(strip=True).replace("(", "").replace(")", "")
            if len(p) > 2 else ""
        )
        score_str = f"📊 {runs}-{wickets} ({overs} overs)"

        event_text = ""
        status_div = soup.find(
            "div",
            class_=lambda x: x and any(c in x for c in ["text-cb-danger", "text-cb-info", "text-cb-success"]),
        )
        if status_div:
            event_text = status_div.get_text(strip=True)

        if is_result_text(event_text):
            return f"{score_str}\n🎯 *Result:* {event_text}"
        return f"{score_str}\n🔥 *Latest:* {event_text}" if event_text else score_str
    except Exception as exc:
        logger.warning("instant score failed for %s: %s", match_url, exc)
        return "Error loading score"


# =====================
# TOSS NOTIFICATION
# =====================
def fetch_toss_update(match_url, match_name):
    """
    Detect and send the toss notification exactly once per match.
    Persists toss_done=1 to DB so the notification is never re-sent after
    a bot restart.
    """
    m_id   = match_url.split("/")[-2] if "/" in match_url else stable_event_suffix(match_name)
    team_a, team_b = get_teams_from_name(match_name)

    # Persistent guard (survives restarts)
    try:
        db_row = cursor.execute("SELECT toss_done FROM state WHERE m_id=?", (m_id,)).fetchone()
        if db_row and db_row[0] == 1:
            return
    except Exception:
        pass

    # In-memory fast path (avoids a DB hit every 15 s within a session)
    if match_url not in match_state:
        match_state[match_url] = {"toss_sent": False}
    if match_state[match_url]["toss_sent"]:
        return

    # Mobile scorecard page reliably surfaces the toss <div>
    scorecard_url = (
        match_url
        .replace("live-cricket-scores",  "live-cricket-scorecard")
        .replace("cricket-scores",       "live-cricket-scorecard")
        .replace("www.cricbuzz.com",      "m.cricbuzz.com")
    )

    try:
        response = requests.get(scorecard_url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return
        soup = BeautifulSoup(response.text, "html.parser")

        toss_label = soup.find(
            lambda tag: tag.name == "div"
            and "font-bold" in tag.get("class", [])
            and "Toss" in tag.get_text()
        )
        if not toss_label:
            return
        toss_text = toss_label.find_next("div").get_text(strip=True)

        match_state[match_url]["toss_sent"] = True

        try:
            cursor.execute(
                """INSERT INTO state
                   (m_id, last_over, last_wickets, toss_done, last_wicket_over,
                    innings, last_double_strike_wk, last_score)
                   VALUES (?, 0, 0, 1, -10.0, 1, 0, 0)
                   ON CONFLICT(m_id) DO UPDATE SET toss_done = 1""",
                (m_id,),
            )
            conn.commit()
        except Exception as db_exc:
            logger.warning("fetch_toss_update DB persist failed: %s", db_exc)

        heading = f"🪙 *TOSS – {_vs_tag(team_a, team_b)}* 🪙"
        msg = (
            f"{heading}\n"
            f"—————————————————\n"
            f"🏆 *{match_name}*\n\n"
            f"🏟 *{toss_text}*\n\n"
            f"🖼 [Tap for Toss Photos]({get_img_link(match_name + ' Toss')})\n"
            f"—————————————————\n"
            f"🏏 _Match starting soon! Get ready!_"
        )
        mf = {
            "match_name":   match_name,
            "event_type":   "TOSS",
            "status_text":  toss_text,
            "innings":      1,
            "team_batting": team_a,
            "team_bowling": team_b,
        }
        send_telegram(msg, pro_edit=True, match_facts=mf)
    except Exception as exc:
        logger.warning("fetch_toss_update failed: %s", exc)


# =====================
# LIVE TEAM DETECTION
# =====================
def get_live_teams(soup, default_a, default_b):
    """
    Detect the currently batting/bowling teams from Cricbuzz HTML markers.
    Falls back to the supplied defaults if detection fails.
    """
    team_batting = ""
    team_bowling = ""

    try:
        bat_div = soup.find(class_=lambda x: x and "cb-text-bat" in x)
        if bat_div:
            parent = bat_div.find_parent("div", class_="flex")
            if parent:
                team_batting = parent.find("div").get_text(strip=True)

        bowl_div = soup.find(class_=lambda x: x and "cb-text-bowl" in x)
        if bowl_div:
            parent = bowl_div.find_parent("div", class_="flex")
            if parent:
                team_bowling = parent.find("div").get_text(strip=True)
    except Exception:
        pass

    return team_batting or default_a, team_bowling or default_b


# =====================
# MAIN MATCH UPDATE LOOP
# =====================
def fetch_match_update(match_url, match_name):
    """
    Core per-match processing function; called every 15 s per live match.
    Detects and dispatches notifications in priority order:

      ① Match result
      ② Innings complete
      ③ Weather/rain delay
      ④ Test match breaks (stumps / lunch / tea / drinks)
      ⑤ Strategic timeout
      ⑥ Impact player substitution
      ⑦ Big over (18+ runs in a single over)
      ⑧ Batting collapse / Double strike
      ⑨ Over milestones (powerplay, 10-over, 15-over, death overs …)
      ⑩ Player milestones (50, 100, orange/purple cap)
    """
    try:
        response = requests.get(match_url, headers=HEADERS, timeout=15)
        soup     = BeautifulSoup(response.text, "html.parser")
        m_id     = match_url.split("/")[-2] if "/" in match_url else stable_event_suffix(match_name)

        team_a, team_b = get_teams_from_name(match_name)

        # ── STATUS TEXT ──────────────────────────────────────────────────
        status_text = ""
        status_div  = soup.find(
            "div",
            class_=lambda x: x and any(c in x for c in [
                "text-cb-danger", "text-cb-info", "text-cb-success",
                "cb-text-complete", "cb-text-abandon",
            ]),
        )
        if status_div:
            status_text = status_div.get_text(strip=True)

        # Fallback: keyword scan when the primary class-based lookup fails
        if not status_text:
            alt_status = soup.find(
                lambda tag: tag.name == "div"
                and tag.get("class")
                and any(phrase in tag.get_text(strip=True).lower() for phrase in [
                    "won by", "abandoned", "target ", "innings break", "stumps", "no result",
                ])
            )
            if alt_status and len(alt_status.get_text(strip=True)) < 100:
                status_text = alt_status.get_text(strip=True)

        status_lower  = status_text.lower()
        is_match_over = is_result_text(status_lower)

        # ── LOAD STATE FROM DB ───────────────────────────────────────────
        try:
            row = cursor.execute(
                "SELECT last_over, last_wickets, toss_done, last_wicket_over, "
                "innings, last_double_strike_wk, last_score FROM state WHERE m_id=?",
                (m_id,),
            ).fetchone()
            if row:
                last_ov, last_wk, toss_done, last_wk_ov, current_innings, last_double_strike_wk, last_score = row
            else:
                last_ov, last_wk, toss_done, last_wk_ov, current_innings, last_double_strike_wk, last_score = (
                    0.0, 0, 0, -10.0, 1, 0, 0
                )
        except Exception:
            last_ov, last_wk, toss_done, last_wk_ov, current_innings, last_double_strike_wk, last_score = (
                0.0, 0, 0, -10.0, 1, 0, 0
            )

        # ── SCORE PARSING ────────────────────────────────────────────────
        score_div = soup.find(
            "div",
            class_=lambda x: x and (("text-3xl" in x and "font-bold" in x) or "cb-font-20" in x),
        )
        runs, wickets   = 0, 0
        overs_raw       = ""
        cur_overs       = 0.0
        cur_balls       = 0
        full_score_text = ""

        if score_div:
            full_score_text = score_div.get_text(separator=" ", strip=True)
            p = score_div.find_all("div")
            if p:
                runs_text = p[0].get_text(strip=True).replace(",", "")
                runs      = int("".join(filter(str.isdigit, runs_text)) or 0)
                if len(p) > 1:
                    w_text  = p[1].get_text(strip=True).replace("-", "").replace("/", "")
                    wickets = int(w_text) if w_text.isdigit() else 0
                if len(p) > 2:
                    overs_raw = p[2].get_text(strip=True).replace("(", "").replace(")", "")

                # parse_overs() handles "14.3 Ov", "(14.3)", plain "14.3", etc.
                cur_overs = parse_overs(overs_raw)
                cur_balls = overs_to_balls(overs_raw)

        # ── INNINGS DETECTION ────────────────────────────────────────────
        # If overs counter reset significantly → second innings has started
        if cur_overs < last_ov - 5:
            last_ov               = 0.0
            last_wk               = 0
            last_wk_ov            = -10.0
            last_double_strike_wk = 0
            last_score            = 0
            current_innings       = 2

        # Swap batting/bowling fallback defaults for the 2nd innings
        if current_innings == 2:
            default_bat, default_bowl = team_b, team_a
        else:
            default_bat, default_bowl = team_a, team_b

        team_batting, team_bowling = get_live_teams(soup, default_bat, default_bowl)
        score_display = f"{team_batting} {runs}/{wickets}" if team_batting else f"{runs}/{wickets}"

        # BUG FIX: guard "target" with current_innings == 1 to prevent this
        # phrase re-triggering an innings-break notification once innings 2 begins.
        is_innings_break = (
            (wickets == 10 and not is_match_over)
            or "innings break" in status_lower
            or ("target" in status_lower and current_innings == 1)
        )
        # Test day/session breaks are NOT innings completions
        is_test_break = any(phrase in status_lower for phrase in ["stumps", "lunch", "tea", "drinks"])

        # ── COMMENTARY ───────────────────────────────────────────────────
        commentary_text = ""
        cm = soup.find("div", class_=lambda x: x and "leading-6" in x)
        if cm:
            eb = cm.find_all("div", recursive=False)
            if eb:
                t  = eb[0] if "." in overs_raw else eb[-1]
                fl = t.find("div", class_=lambda x: x and "flex" in x and "gap-4" in x)
                if fl:
                    event_divs = fl.find_all("div", recursive=False)
                    if len(event_divs) >= 2:
                        commentary_text = event_divs[1].get_text(strip=True)

        event_text  = status_text if status_text else commentary_text
        event_lower = event_text.lower()

        # Shared dict sent to Groq for every AI-polished message this cycle
        match_facts = {
            "match_name":    match_name,
            "event_type":    "LIVE UPDATE",
            "team_batting":  team_batting,
            "team_bowling":  team_bowling,
            "innings":       current_innings,
            "score_display": score_display,
            "status_text":   status_text,
            "wickets":       wickets,
            "raw_data":      full_score_text + " " + commentary_text,
        }

        messages_to_send = []

        # ══════════════════════════════════════════════════════════════════
        # 🏆 MATCH END — highest priority; sends immediately and exits
        # ══════════════════════════════════════════════════════════════════
        if is_match_over:
            eid = f"{m_id}_MATCH_END"
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                match_facts["event_type"] = "MATCH_END"
                heading = f"🏆 *FINAL RESULT – {_vs_tag(team_a, team_b)}* 🏆"
                msg = (
                    f"{heading}\n"
                    f"—————————————————\n"
                    f"🎯 *{status_text}*\n\n"
                    f"🔹 {match_name}\n"
                    f"🔹 Final Score: *{score_display}* ({overs_raw})\n\n"
                    f"🖼 [Tap for Winning Moments]({get_img_link(match_name)})\n"
                    f"—————————————————\n"
                    f"✅ *Coverage concluded.*"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                cursor.execute("INSERT OR REPLACE INTO tracking_config VALUES (?, ?, 0)", (m_id, match_name))
                conn.commit()
                send_telegram(msg, pro_edit=True, match_facts=match_facts)
                return  # Stop all further processing for this match

        # ══════════════════════════════════════════════════════════════════
        # 🛑 INNINGS COMPLETE
        # EID uses current_innings (not runs) — prevents re-firing each poll
        # cycle when Cricbuzz keeps "Target: X" in the status banner.
        # ══════════════════════════════════════════════════════════════════
        if is_innings_break:
            eid = f"{m_id}_INNINGS_BREAK_{current_innings}"
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                match_facts["event_type"] = "INNINGS_BREAK"
                heading = f"🛑 *INNINGS COMPLETE – {_abbr(team_batting)} {runs}/{wickets}* 🛑"
                msg = (
                    f"{heading}\n"
                    f"—————————————————\n"
                    f"🏏 *{match_name}*\n\n"
                    f"📊 *FINAL SCORE:* *{score_display}*\n"
                    f"🎯 *UPDATE:* _{status_text}_\n\n"
                    f"🖼 [Tap for Match Gallery]({get_img_link(match_name)})\n"
                    f"—————————————————\n"
                    f"🕒 _Second innings starts shortly._"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                messages_to_send.append((msg, match_facts.copy()))

        # ══════════════════════════════════════════════════════════════════
        # 🌦 WEATHER / RAIN DELAY
        # ══════════════════════════════════════════════════════════════════
        if any(x in status_lower for x in ["rain", "drizzle", "interrupted", "delayed", "covers"]):
            eid = f"{m_id}_RAIN_{stable_event_suffix(status_text)}"
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                match_facts["event_type"] = "WEATHER_DELAY"
                heading = f"🌦 *WEATHER DELAY – {_vs_tag(team_a, team_b)}* 🌦"
                msg = (
                    f"{heading}\n"
                    f"—————————————————\n"
                    f"⚠️ {status_text}\n\n"
                    f"🕒 Match currently interrupted. Stay tuned for restart updates!"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                messages_to_send.append((msg, match_facts.copy()))

        # ══════════════════════════════════════════════════════════════════
        # ⏸️ TEST MATCH BREAKS (stumps / lunch / tea / drinks)
        # ══════════════════════════════════════════════════════════════════
        if is_test_break:
            eid = f"{m_id}_BREAK_{stable_event_suffix(status_text)}"
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                break_icons = {"stumps": "🌙", "lunch": "🍽️", "tea": "☕", "drinks": "💧"}
                break_key   = next((k for k in break_icons if k in status_lower), "break")
                break_icon  = break_icons.get(break_key, "⏸️")
                match_facts["event_type"] = "TEST_BREAK"
                heading = f"{break_icon} *{break_key.upper()} – {_vs_tag(team_a, team_b)}* {break_icon}"
                msg = (
                    f"{heading}\n"
                    f"—————————————————\n"
                    f"🏏 *{match_name}*\n\n"
                    f"📊 *SCORE:* *{score_display}* ({overs_raw})\n"
                    f"⏸️ _{status_text}_\n"
                    f"—————————————————\n"
                    f"🔔 _Play resumes shortly._"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                messages_to_send.append((msg, match_facts.copy()))

        # ══════════════════════════════════════════════════════════════════
        # ⏸️ STRATEGIC TIMEOUT
        # ══════════════════════════════════════════════════════════════════
        if "timeout" in event_lower and "strategic" in event_lower:
            eid = f"{m_id}_TIMEOUT_{int(cur_overs)}"
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                match_facts["event_type"] = "STRATEGIC TIMEOUT"
                heading = f"⏸️ *STRATEGIC TIMEOUT – {_vs_tag(team_a, team_b)}* ⏸️"
                msg = (
                    f"{heading}\n"
                    f"—————————————————\n"
                    f"🏏 *MATCH:* {match_name}\n"
                    f"📊 *SCORE:* *{score_display}* ({overs_raw})\n\n"
                    f"Time to rethink strategies! Who is winning this phase?\n"
                    f"—————————————————\n"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                messages_to_send.append((msg, match_facts.copy()))

        # ══════════════════════════════════════════════════════════════════
        # 🔄 IMPACT PLAYER SUBSTITUTION
        # ══════════════════════════════════════════════════════════════════
        if any(x in event_lower for x in ["impact player", "impact sub", "substituted by"]):
            eid = f"{m_id}_IMPACT_{stable_event_suffix(event_text)}"
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                match_facts["event_type"] = "IMPACT PLAYER SUBSTITUTION"
                heading = f"🔄 *IMPACT SUB – {_vs_tag(team_a, team_b)}* 🔄"
                msg = (
                    f"{heading}\n"
                    f"—————————————————\n"
                    f"🏏 *MATCH:* {match_name}\n\n"
                    f"📢 _{event_text}_\n\n"
                    f"A major tactical move! Let's see how this pays off.\n"
                    f"—————————————————\n"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                messages_to_send.append((msg, match_facts.copy()))

        # ══════════════════════════════════════════════════════════════════
        # 🔥 BIG OVER (18+ runs in a single completed over)
        # ══════════════════════════════════════════════════════════════════
        if cur_overs > last_ov and str(cur_overs).endswith(".0"):
            runs_this_over = runs - last_score
            if runs_this_over >= 18:
                eid = f"{m_id}_BIG_OVER_{int(cur_overs)}_{runs_this_over}"
                if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                    match_facts["event_type"] = f"BIG OVER: {runs_this_over} runs"
                    heading = f"🔥 *{runs_this_over} OFF ONE OVER – {_abbr(team_batting)} GO LARGE* 🔥"
                    msg = (
                        f"{heading}\n"
                        f"—————————————————\n"
                        f"🏆 *{match_name}*\n"
                        f"📊 *SCORE:* *{score_display}* ({overs_raw})\n\n"
                        f"Momentum shifted completely! 🚀\n"
                        f"—————————————————\n"
                    )
                    cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                    messages_to_send.append((msg, match_facts.copy()))
            last_score = runs  # Always update after each completed over

        # ══════════════════════════════════════════════════════════════════
        # 🎯 WICKET EVENTS
        # ══════════════════════════════════════════════════════════════════
        if wickets > last_wk:
            new_wk_ov = cur_overs

            # Early collapse: 3 wickets in the first 6 overs
            # EID includes current_innings so both innings can trigger independently
            if wickets == 3 and cur_overs <= 6.0 and last_wk < 3:
                eid = f"{m_id}_COLLAPSE_3WK_{current_innings}"
                if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                    match_facts["event_type"] = "BATTING_COLLAPSE"
                    heading = f"🚨 *EARLY COLLAPSE – {_abbr(team_batting)} IN TROUBLE* 🚨"
                    msg = (
                        f"{heading}\n"
                        f"—————————————————\n"
                        f"🏏 *MATCH:* {match_name}\n"
                        f"📊 *SCORE:* *{score_display}* ({overs_raw})\n"
                        f"💬 *LATEST WICKET:* _{event_text}_\n\n"
                        f"🖼 [Tap for Match Action]({get_img_link(match_name)})\n"
                        f"—————————————————\n"
                        f"📉 *The batting side is under massive pressure!*"
                    )
                    cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                    messages_to_send.append((msg, match_facts.copy()))

            # Double strike: 2 wickets within ~1 over
            # last_wk_ov >= 0 guards against the sentinel -10.0 (no prior wicket).
            elif (
                last_wk_ov >= 0
                and abs(cur_balls - overs_to_balls(last_wk_ov)) <= 6
                and wickets >= last_double_strike_wk + 2
            ):
                eid = f"{m_id}_DOUBLE_STRIKE_{wickets}"
                if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                    match_facts["event_type"] = "DOUBLE_WICKET"
                    heading = f"🎯 *DOUBLE STRIKE – {_abbr(team_bowling)} ON FIRE* 🎯"
                    msg = (
                        f"{heading}\n"
                        f"—————————————————\n"
                        f"🏏 *MATCH:* {match_name}\n"
                        f"📊 *NEW SCORE:* *{score_display}* ({overs_raw})\n"
                        f"💬 *LATEST:* _{event_text}_\n\n"
                        f"🖼 [Tap for Celebration Photos]({get_img_link(match_name)})\n"
                        f"—————————————————\n"
                        f"⚠️ *Huge turning point in the game!*"
                    )
                    cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                    last_double_strike_wk = wickets
                    messages_to_send.append((msg, match_facts.copy()))

            last_wk_ov = new_wk_ov
            last_wk    = wickets  # ← BUG FIX: keep in sync so next poll doesn't re-fire

        # ══════════════════════════════════════════════════════════════════
        # 📊 OVER MILESTONES
        # T20: powerplay(6), 10, 15, 20 | ODI: 10,20,30,40,50 | Test: 10-90
        # ══════════════════════════════════════════════════════════════════
        is_t20 = (
            "T20" in match_name.upper()
            or "INDIAN PREMIER LEAGUE" in match_name.upper()
            or " IPL " in match_name.upper()
            or match_name.upper().endswith(" IPL")
        )
        is_odi = "ODI" in match_name.upper()

        if is_t20:
            milestones = [6, 10, 15, 20]
        elif is_odi:
            milestones = [10, 20, 30, 40, 50]
        else:
            milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        passed_m      = None
        last_ov_balls = overs_to_balls(last_ov)
        for m in milestones:
            m_balls = m * 6
            if last_ov_balls < m_balls <= cur_balls:
                passed_m = m
                break

        if passed_m:
            eid = f"{m_id}_OV_{passed_m}_{current_innings}"  # ← BUG FIX: removed runs from EID
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                crr = f"{(runs / cur_overs):.2f}" if cur_overs else "N/A"

                phase_header = f"{passed_m}-OVER"
                event_tag    = f"{phase_header} SUMMARY"
                if is_t20 and passed_m == 6:
                    phase_header = "POWERPLAY END"
                    event_tag    = "POWERPLAY"
                elif is_t20 and passed_m in [15, 20]:
                    phase_header = "DEATH OVERS"

                match_facts["event_type"] = event_tag
                heading = f"🏏 *{phase_header} – {_vs_tag(team_a, team_b)}* 🏏"
                msg = (
                    f"{heading}\n"
                    f"—————————————————\n"
                    f"🏆 *{match_name}*\n\n"
                    f"📊 *SCORE:* *{score_display}*\n"
                    f"🕒 *OVERS:* {cur_overs}\n"
                    f"📈 *RUN RATE:* {crr}\n\n"
                    f"⚡ *LATEST:* _{event_text}_\n\n"
                    f"🖼 [Tap for Match Photos]({get_img_link(match_name)})\n"
                    f"—————————————————\n"
                    f"🔔 *Stay tuned for more live action!*"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                messages_to_send.append((msg, match_facts.copy()))

        # ══════════════════════════════════════════════════════════════════
        # ⭐ PLAYER MILESTONES (50, 100, orange cap, purple cap)
        # ══════════════════════════════════════════════════════════════════
        event_type  = None
        speed_alert = ""
        balls_faced = 999
        ball_match  = re.search(r"(\d+)\s*(balls|b)", event_lower)
        if ball_match:
            balls_faced = int(ball_match.group(1))

        if "orange cap" in event_lower:
            event_type  = "ORANGE CAP"
            speed_alert = "🧢 LEADERBOARD SHAKEUP 🧢\n"
        elif "purple cap" in event_lower:
            event_type  = "PURPLE CAP"
            speed_alert = "🧢 LEADERBOARD SHAKEUP 🧢\n"
        elif any(x in event_lower for x in ["fifty", "half-century", "half century", "50 runs", "reaches 50"]):
            event_type  = "50"
            if balls_faced <= 25:
                speed_alert = "⚡ EXPLOSIVE INNINGS ⚡\n"
        elif any(x in event_lower for x in ["century", "hundred", "100 runs", "reaches 100"]):
            event_type  = "100"
            if balls_faced <= 50:
                speed_alert = "⚡ SENSATIONAL CENTURY ⚡\n"

        if event_type:
            eid = f"{m_id}_MILESTONE_{stable_event_suffix(event_text)}"
            if not cursor.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
                if "CAP" in event_type:
                    header = speed_alert
                else:
                    milestone_heading = f"⭐ *{event_type} MILESTONE – {_abbr(team_batting)}* ⭐"
                    header = (speed_alert + milestone_heading) if speed_alert else milestone_heading

                match_facts["event_type"] = f"PLAYER {event_type} MILESTONE"
                msg = (
                    f"{header}\n"
                    f"—————————————————\n"
                    f"🏏 *MATCH:* {match_name}\n"
                    f"📊 *CURRENT SCORE:* *{score_display}* ({overs_raw})\n"
                    f"💬 *COMMENTARY:* _{event_text}_\n\n"
                    f"🖼 [Tap for Player Photos]({get_img_link(match_name + ' ' + event_text)})\n"
                    f"—————————————————\n"
                    f"👏 *What a moment! Share the news!*"
                )
                cursor.execute("INSERT INTO events VALUES (?)", (eid,))
                messages_to_send.append((msg, match_facts.copy()))

        # ── Send all accumulated messages for this cycle ──
        for m, f in messages_to_send:
            send_telegram(m, pro_edit=True, match_facts=f)

        # ── Persist updated state to DB ──
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO state "
                "(m_id, last_over, last_wickets, toss_done, last_wicket_over, "
                "innings, last_double_strike_wk, last_score) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (m_id, cur_overs, wickets, toss_done, last_wk_ov,
                 current_innings, last_double_strike_wk, last_score),
            )
            conn.commit()
        except sqlite3.Error:
            pass

    except Exception:
        logger.error("fetch_match_update failed for %s:\n%s", match_url, traceback.format_exc())


# =====================
# MAIN LOOP
# =====================
def run_bot():
    """Entry point — runs the polling loop forever with a 15-second sleep."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.error("Missing BOT_TOKEN and/or CHAT_ID. Bot cannot start.")
        return

    logger.info("=" * 50)
    logger.info("BUILD: groq-sdk-v2 | groq_client_ready=%s", groq_client is not None)
    logger.info("=" * 50)
    logger.info("🚀 Cricket Notification Bot Starting...")
    send_telegram(
        "✅ *Live-Only Tracker Active!* 🏏\n"
        "- Zero spam guaranteed.\n"
        "- Use /stop to kill unwanted matches."
    )

    while True:
        try:
            handle_commands()
            handle_daily_briefing()

            matches = scrape_match_links()
            for name, link in matches:
                m_id = link.split("/")[-2]

                row = cursor.execute(
                    "SELECT is_active FROM tracking_config WHERE m_id=?", (m_id,)
                ).fetchone()
                default_active = 0 if is_womens_match(name) else 1
                is_tracking    = row[0] if row else default_active

                if is_tracking == 0:
                    continue

                fetch_toss_update(link, name)
                fetch_match_update(link, name)

        except Exception:
            logger.error("Main Loop Error:\n%s", traceback.format_exc())

        time.sleep(15)


if __name__ == "__main__":
    run_bot()
