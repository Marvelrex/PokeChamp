# check_ou_teams.py
from pathlib import Path
import re

TEAM_DIR      = Path("poke_env/data/static/teams")
BAN_LIST_OU   = {
    "roaringmoon", "fluttermane", "chienpao", "koraidon", "miraidon",
    "archaludon", "ursalunabloodmoon", "calyrexshadow", "calyrexice",
    "regieleki", "zamazentacrowned", "zacian", "zaciancrowned",
}
# If you maintain your own copy of the tier list, adjust the set above.

id_re = re.compile(r"^[A-Za-z0-9' \-]+")       # capture mon name before " @ ..." etc.

def name_to_id(line: str) -> str:
    """
    Convert a raw nickname line ('Great Tusk @ Rocky Helmet') to a
    lowercase Smogon-style id ('greattusk').
    """
    raw_name = id_re.match(line).group(0)
    return re.sub(r"[^A-Za-z0-9]", "", raw_name).lower()

def check_team(path: Path) -> list[str]:
    """
    Return a list of banned mons found in the file, or [] if legal.
    """
    banned_seen = []
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "EVs:", "IVs:", "Ability:", "- ", "Shiny")):
            continue
        mon_id = name_to_id(line)
        if mon_id in BAN_LIST_OU:
            banned_seen.append(mon_id)
        if len(banned_seen) == 6:     # team max size
            break
    return banned_seen

def main():
    any_illegal = False
    for p in sorted(TEAM_DIR.glob("*.txt")):
        banned_found = check_team(p)
        if banned_found:
            any_illegal = True
            print(f"✗ {p.name}: {', '.join(banned_found)}")
        else:
            print(f"✓ {p.name}: OK")
    if not any_illegal:
        print("\nAll teams are legal for Gen 9 OU!")

if __name__ == "__main__":
    main()
