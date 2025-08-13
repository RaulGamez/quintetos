# streamlit_app.py
# -------------------------------------------------------------
# Gestor de quintetos FBCV (norma pasarela) con Firestore (DB)
# -------------------------------------------------------------
import time, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import streamlit as st

# ============== Firestore (Google Cloud) ==============
try:
    from google.cloud import firestore
    from google.oauth2 import service_account
except Exception:
    st.error("Falta el paquete google-cloud-firestore. Añade 'google-cloud-firestore' a requirements.txt.")
    st.stop()

st.set_page_config(page_title="Quintetos FBCV - Norma Pasarela", layout="wide")

# Carga de credenciales desde Secrets (JSON de cuenta de servicio)
if "gcp_service_account" not in st.secrets:
    st.error("Añade tu JSON de cuenta de servicio a Secrets como 'gcp_service_account'.")
    st.stop()

creds = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
db = firestore.Client(project=st.secrets["gcp_service_account"]["project_id"], credentials=creds)

# ===========================
# Reglas Pasarela (FBCV)
# ===========================
CATEGORIES = {
    "INFANTIL FEMENINO": {"periods": 8, "constrained_periods": 6},
    "INFANTIL MASCULINO": {"periods": 8, "constrained_periods": 6},
    "PRE-INFANTIL": {"periods": 8, "constrained_periods": 6},
    "ALEVÍN": {"periods": 6, "constrained_periods": 5},
    "BENJAMÍN": {"periods": 6, "constrained_periods": 5},
}

# ===========================
# Modelos en memoria
# ===========================
@dataclass
class Player:
    id: str
    name: str
    rating: int = 3  # 1-5

@dataclass
class Team:
    id: str
    name: str
    category: str
    players: Dict[str, Player] = field(default_factory=dict)
    rules: Dict = field(default_factory=lambda: {"together": [], "apart": [], "strict": False})

@dataclass
class Lineup:
    period: int
    players: List[str]
    def signature(self) -> str:
        return ",".join(sorted(self.players))

@dataclass
class Schedule:
    category: str
    periods: int
    constrained_periods: int
    lineups: Dict[int, Lineup] = field(default_factory=dict)

# ===========================
# Helpers DB (Firestore)
# ===========================
def _hash_password(pw: str) -> str:
    import hashlib
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

# Estructura de colecciones:
# - users (doc id = username) -> {username, password_hash, created_at}
# - teams (doc id = team_id)  -> {id, owner, name, category, created_at}
# - players (doc id = player_id) -> {id, team_id, name, rating, created_at}
# - rules (doc id = team_id)  -> {team_id, strict, updated_at}
# - rules_together (autoid)   -> {team_id, a, b, created_at}
# - rules_apart (autoid)      -> {team_id, a, b, created_at}
# - ratings (autoid)          -> {team_id, signature, rating, notes, created_at}

# ---- USERS ----
def db_user_get(username: str) -> Optional[dict]:
    doc = db.collection("users").document(username).get()
    return doc.to_dict() if doc.exists else None

def db_user_create(username: str, password_hash: str):
    db.collection("users").document(username).set({
        "username": username,
        "password_hash": password_hash,
        "created_at": firestore.SERVER_TIMESTAMP
    })

# ---- TEAMS ----
def db_teams_by_owner(owner: str) -> List[dict]:
    q = db.collection("teams").where("owner", "==", owner).order_by("created_at").stream()
    return [dict(doc.to_dict()) for doc in q]

def db_team_get(team_id: str) -> Optional[dict]:
    doc = db.collection("teams").document(team_id).get()
    return doc.to_dict() if doc.exists else None

def db_team_create(owner: str, name: str, category: str) -> dict:
    team_id = f"{owner}:{int(time.time()*1000)}"
    db.collection("teams").document(team_id).set({
        "id": team_id, "owner": owner, "name": name, "category": category,
        "created_at": firestore.SERVER_TIMESTAMP
    })
    db.collection("rules").document(team_id).set({
        "team_id": team_id, "strict": False, "updated_at": firestore.SERVER_TIMESTAMP
    })
    return {"id": team_id, "owner": owner, "name": name, "category": category}

def db_team_update(team_id: str, name: str, category: str):
    db.collection("teams").document(team_id).update({"name": name, "category": category})

# ---- PLAYERS ----
def db_players_by_team(team_id: str) -> List[dict]:
    q = db.collection("players").where("team_id", "==", team_id).order_by("created_at").stream()
    return [dict(doc.to_dict()) for doc in q]

def db_player_add(team_id: str, name: str, rating: int) -> dict:
    pid = f"{int(time.time()*1000)}-{random.randint(100,999)}"
    db.collection("players").document(pid).set({
        "id": pid, "team_id": team_id, "name": name, "rating": int(rating),
        "created_at": firestore.SERVER_TIMESTAMP
    })
    return {"id": pid, "team_id": team_id, "name": name, "rating": int(rating)}

def db_player_update(pid: str, name: str, rating: int):
    db.collection("players").document(pid).update({"name": name, "rating": int(rating)})

def db_player_delete(pid: str):
    db.collection("players").document(pid).delete()

# ---- RULES ----
def db_rules_get(team_id: str) -> Dict:
    doc = db.collection("rules").document(team_id).get()
    strict = bool(doc.to_dict().get("strict")) if doc.exists else False
    tog_q = db.collection("rules_together").where("team_id", "==", team_id).stream()
    apt_q = db.collection("rules_apart").where("team_id", "==", team_id).stream()
    together = [[d.to_dict()["a"], d.to_dict()["b"]] for d in tog_q]
    apart = [[d.to_dict()["a"], d.to_dict()["b"]] for d in apt_q]
    # normaliza orden a<b
    together = [sorted(p) for p in together if len(p) == 2]
    apart = [sorted(p) for p in apart if len(p) == 2]
    return {"strict": strict, "together": together, "apart": apart}

def db_rules_set_strict(team_id: str, strict: bool):
    db.collection("rules").document(team_id).set({
        "team_id": team_id, "strict": bool(strict), "updated_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

def db_rules_add_pairs(team_id: str, table: str, pairs: List[List[str]]):
    col = "rules_together" if table == "rules_together" else "rules_apart"
    batch = db.batch()
    for a, b in pairs:
        a2, b2 = sorted([a, b])
        ref = db.collection(col).document()
        batch.set(ref, {"team_id": team_id, "a": a2, "b": b2, "created_at": firestore.SERVER_TIMESTAMP})
    if pairs:
        batch.commit()

def db_rules_delete_pair(team_id: str, table: str, a: str, b: str):
    col = "rules_together" if table == "rules_together" else "rules_apart"
    a2, b2 = sorted([a, b])
    q = db.collection(col).where("team_id","==",team_id).where("a","==",a2).where("b","==",b2).stream()
    for doc in q:
        doc.reference.delete()

# ---- RATINGS ----
def db_rate_lineup(team_id: str, signature: str, rating: int, notes: str = ""):
    db.collection("ratings").document().set({
        "team_id": team_id, "signature": signature, "rating": int(rating), "notes": notes,
        "created_at": firestore.SERVER_TIMESTAMP
    })

def db_top_lineups(team_id: str, top_k: int = 10) -> List[List[str]]:
    # Promedia en cliente (sencillo y suficiente para los volúmenes esperados)
    rows = [d.to_dict() for d in db.collection("ratings").where("team_id","==",team_id).stream()]
    from collections import defaultdict
    acc = defaultdict(lambda: {"sum":0, "n":0})
    for r in rows:
        sig = r.get("signature","")
        if not sig: continue
        acc[sig]["sum"] += int(r.get("rating",0))
        acc[sig]["n"] += 1
    avg_list = [(v["sum"]/v["n"], s) for s, v in acc.items() if v["n"]>0]
    avg_list.sort(reverse=True, key=lambda x: x[0])
    top = [sig for _, sig in avg_list[:top_k]]
    return [sig.split(",") for sig in top]

# ===========================
# Autenticación simple
# ===========================
def register_user(username: str, password: str):
    if not username or not password:
        return False, "Usuario y contraseña son obligatorios."
    if db_user_get(username):
        return False, "El usuario ya existe."
    db_user_create(username, _hash_password(password))
    return True, "Registro completado."

def login_user(username: str, password: str):
    row = db_user_get(username)
    if not row:
        return False, "Usuario no encontrado."
    if row.get("password_hash") != _hash_password(password):
        return False, "Contraseña incorrecta."
    return True, "Bienvenido/a."

# ===========================
# Carga/guardado de Team completo
# ===========================
def load_team(team_id: str) -> Optional[Team]:
    raw = db_team_get(team_id)
    if not raw:
        return None
    players = {}
    for pr in db_players_by_team(team_id):
        players[pr["id"]] = Player(id=pr["id"], name=pr["name"], rating=int(pr["rating"]))
    rules = db_rules_get(team_id)
    return Team(id=raw["id"], name=raw["name"], category=raw["category"], players=players, rules=rules)

def save_team_meta(team: Team):
    db_team_update(team.id, team.name, team.category)

# ===========================
# Reglas personalizadas (helpers)
# ===========================
def _normalize_pair(a: str, b: str) -> List[str]:
    if a == b:
        return []
    a2, b2 = sorted([a, b])
    return [a2, b2]

def _has_pair(pairs: List[List[str]], pair: List[str]) -> bool:
    return any(sorted(p) == pair for p in pairs)

def violates_rules(players_in_lineup: List[str], team: Team) -> Tuple[bool, List[str], List[str]]:
    rules = team.rules or {}
    together = rules.get("together", [])
    apart = rules.get("apart", [])
    strict = bool(rules.get("strict", False))

    apart_hits = []
    for a, b in apart:
        if a in players_in_lineup and b in players_in_lineup:
            apart_hits.append("Regla 'no juntas' incumplida.")

    together_misses = []
    if strict:
        for a, b in together:
            if (a in players_in_lineup) ^ (b in players_in_lineup):
                together_misses.append("Regla 'sí juntas' incumplida.")

    if strict:
        violates = bool(apart_hits or together_misses)
        return violates, [], []
    else:
        # solo avisos suaves de 'no juntas' en modo no estricto
        return False, apart_hits, []

# ===========================
# Validación (pasarela + equilibrio + reglas)
# ===========================
def validate_schedule(team: Team, schedule: Schedule) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    # 1) Quintetos definidos y sin duplicados
    for p in range(1, schedule.periods + 1):
        l = schedule.lineups.get(p)
        if not l:
            errors.append(f"Periodo {p}: no hay quinteto definido.")
            continue
        if len(l.players) != 5:
            errors.append(f"Periodo {p}: el quinteto debe tener 5 jugadores/as (actual: {len(l.players)}).")
        if len(set(l.players)) != len(l.players):
            errors.append(f"Periodo {p}: hay jugadores/as repetidos en el quinteto.")

    # 2) Pertenencia al equipo
    all_ids = set(team.players.keys())
    for p, l in schedule.lineups.items():
        for pid in l.players:
            if pid not in all_ids:
                errors.append(f"Periodo {p}: jugador/a no pertenece al equipo.")

    # 3) Pasarela en ventana restringida
    constr = schedule.constrained_periods
    play_counts = {pid: 0 for pid in all_ids}
    bench_counts = {pid: 0 for pid in all_ids}
    for p in range(1, constr + 1):
        l = schedule.lineups.get(p)
        on_court = set(l.players) if l else set()
        for pid in all_ids:
            if pid in on_court:
                play_counts[pid] += 1
            else:
                bench_counts[pid] += 1
    for pid in all_ids:
        if play_counts[pid] < 2:
            errors.append(f"{team.players[pid].name}: juega {play_counts[pid]} periodos en los primeros {constr} (mín. 2).")
        if bench_counts[pid] < 2:
            errors.append(f"{team.players[pid].name}: descansa {bench_counts[pid]} periodos en los primeros {constr} (mín. 2).")

    # 3b) Prohibir 3 seguidos en P1–P6
    window6 = min(6, schedule.periods)
    for pid in all_ids:
        consec = 0
        for p in range(1, window6 + 1):
            l = schedule.lineups.get(p)
            on = l and (pid in l.players)
            if on:
                consec += 1
                if consec >= 3:
                    errors.append(f"{team.players[pid].name}: juega 3 periodos seguidos dentro de los primeros {window6}.")
                    break
            else:
                consec = 0

    # 4) Reglas por periodo
    for p in range(1, schedule.periods + 1):
        l = schedule.lineups.get(p)
        if not l:
            continue
        hard, soft_apart, _ = violates_rules(l.players, team)
        if hard:
            errors.append(f"Periodo {p}: violación de reglas personalizadas.")
        else:
            warnings.extend([f"Periodo {p}: {m}" for m in soft_apart])

    # 5) Equilibrio por puntuación
    ratings = {pid: team.players[pid].rating for pid in all_ids}
    if schedule.lineups and ratings:
        team_avg = sum(ratings.values()) / len(ratings)
        target = 5 * team_avg
        tolerance = 2.5
        for p, l in schedule.lineups.items():
            score = sum(ratings.get(pid, 3) for pid in l.players)
            if abs(score - target) > tolerance:
                warnings.append(f"Periodo {p}: puntuación {score:.1f} desequilibrada (objetivo ~{target:.1f}±{tolerance}).")

    return {"errors": errors, "warnings": warnings}

# ===========================
# Generación automática
# ===========================
def _period_config_for_category(category: str) -> Tuple[int, int]:
    cfg = CATEGORIES[category]
    return cfg["periods"], cfg["constrained_periods"]

def _compute_targets_by_rating(ratings: Dict[str, int], periods: int) -> Dict[str, int]:
    import math
    total_slots = periods * 5
    s = sum(ratings.values())
    if s == 0:
        base = total_slots // len(ratings)
        targets = {pid: base for pid in ratings}
        remainder = total_slots - base * len(ratings)
        for pid in list(ratings.keys())[:remainder]:
            targets[pid] += 1
        return targets
    raw = {pid: (ratings[pid] / s) * total_slots for pid in ratings}
    flo = {pid: int(math.floor(raw[pid])) for pid in ratings}
    remainder = total_slots - sum(flo.values())
    decimals = sorted(((raw[pid] - flo[pid], pid) for pid in ratings), reverse=True)
    for i in range(remainder):
        _, pid = decimals[i]
        flo[pid] += 1
    for pid in flo:
        flo[pid] = max(1, min(periods, flo[pid]))
    diff = total_slots - sum(flo.values())
    if diff != 0:
        order = sorted(ratings, key=lambda q: ratings[q], reverse=(diff < 0))
        i = 0
        while diff != 0 and i < len(order):
            pid = order[i]
            if diff < 0 and flo[pid] > 1:
                flo[pid] -= 1; diff += 1
            elif diff > 0 and flo[pid] < periods:
                flo[pid] += 1; diff -= 1
            else:
                i += 1
    return flo

def generate_automatic_schedule(team: Team, seed_top_lineups: bool = True, variability: float = 0.15) -> Schedule:
    periods, constr = _period_config_for_category(team.category)
    schedule = Schedule(team.category, periods, constr)
    rng = random.Random(time.time())

    player_ids = list(team.players.keys())
    if len(player_ids) < 5:
        raise ValueError("Se requieren al menos 5 jugadores/as en el equipo.")
    ratings = {pid: team.players[pid].rating for pid in player_ids}

    rules = team.rules or {}
    together_pairs = [sorted(p) for p in rules.get("together", []) if len(p) == 2 and p[0] in team.players and p[1] in team.players]
    apart_pairs = [sorted(p) for p in rules.get("apart", []) if len(p) == 2 and p[0] in team.players and p[1] in team.players]
    strict_rules = bool(rules.get("strict", False))

    def jitter(x: float) -> float:
        return x * (1.0 + variability * (rng.random() - 0.5) * 2)
    def lineup_score(players: List[str]) -> float:
        return sum(ratings[p] for p in players)

    targets_total = _compute_targets_by_rating(ratings, periods)
    team_avg = sum(ratings.values()) / len(ratings)
    target_score = 5 * team_avg
    tolerance = 2.5
    total_play = {pid: 0 for pid in player_ids}

    def creates_three_streak(pid: str, period: int, current_lineups: Dict[int, Lineup]) -> bool:
        if period > min(6, periods):
            return False
        prev1 = current_lineups.get(period - 1)
        prev2 = current_lineups.get(period - 2)
        c1 = prev1 and (pid in prev1.players)
        c2 = prev2 and (pid in prev2.players)
        return bool(c1 and c2)

    def violates_pair_rules_if_add(pid: str, current_players: List[str]) -> bool:
        for a, b in apart_pairs:
            if pid == a and b in current_players: return True
            if pid == b and a in current_players: return True
        if strict_rules:
            for a, b in together_pairs:
                if pid == a and b not in current_players:
                    for x, y in apart_pairs:
                        if (b == x and y in current_players) or (b == y and x in current_players):
                            return True
                if pid == b and a not in current_players:
                    for x, y in apart_pairs:
                        if (a == x and y in current_players) or (a == y and x in current_players):
                            return True
        return False

    def enforce_together_if_needed(lineup: List[str]) -> List[str]:
        if not strict_rules:
            return lineup[:5]
        in_set = set(lineup)
        need = []
        for a, b in together_pairs:
            if (a in in_set) ^ (b in in_set):
                need.append((a, b))
        for a, b in need:
            missing = b if a in in_set else a
            if missing in in_set:
                continue
            if len(lineup) >= 5:
                best_idx, best_gap = None, 1e9
                for i, out_pid in enumerate(lineup):
                    if out_pid in (a, b):
                        continue
                    bad = False
                    for x, y in apart_pairs:
                        if (missing == x and y in (set(lineup) - {out_pid})) or (missing == y and x in (set(lineup) - {out_pid})):
                            bad = True; break
                    if bad: continue
                    trial = lineup.copy(); trial[i] = missing
                    gap = abs(lineup_score(trial) - target_score)
                    if gap < best_gap:
                        best_gap, best_idx = gap, i
                if best_idx is not None:
                    lineup = lineup.copy(); lineup[best_idx] = missing
            else:
                bad = any((missing == x and y in lineup) or (missing == y and x in lineup) for x, y in apart_pairs)
                if not bad:
                    lineup = lineup + [missing]
        return lineup[:5]

    # Semilla: mejores quintetos valorados
    seeded_periods = set()
    if seed_top_lineups:
        top_lineups = db_top_lineups(team.id, top_k=3)
        for i, lp in enumerate(top_lineups, start=1):
            if i > constr:
                break
            if len(lp) == 5 and all(pid in team.players for pid in lp):
                adjusted = []
                for pid in lp:
                    if creates_three_streak(pid, i, schedule.lineups): continue
                    if violates_pair_rules_if_add(pid, adjusted): continue
                    adjusted.append(pid)
                if len(adjusted) < 5:
                    cands = [pid for pid in player_ids if pid not in adjusted]
                    cands.sort(key=lambda q: (targets_total[q] - total_play[q], jitter(ratings[q])), reverse=True)
                    for pid in cands:
                        if len(adjusted) == 5: break
                        if creates_three_streak(pid, i, schedule.lineups): continue
                        if violates_pair_rules_if_add(pid, adjusted): continue
                        adjusted.append(pid)
                adjusted = enforce_together_if_needed(adjusted)
                schedule.lineups[i] = Lineup(i, adjusted[:5])
                for pid in schedule.lineups[i].players:
                    total_play[pid] += 1
                seeded_periods.add(i)

    # Ventana restringida
    window_slots = constr * 5
    n = len(player_ids)
    base_assign = min(2, window_slots // n) if n > 0 else 0
    need_window = {pid: base_assign for pid in player_ids}
    extra = window_slots - base_assign * n
    if extra > 0:
        order = sorted(player_ids, key=lambda q: (ratings[q], rng.random()), reverse=True)
        i = 0
        while extra > 0:
            pid = order[i % len(order)]
            if need_window[pid] < max(0, constr - 2):
                need_window[pid] += 1
                extra -= 1
            i += 1

    # Periodos 1..constr
    for p in range(1, constr + 1):
        if p in seeded_periods:
            continue
        lineup = []
        candidates = list(player_ids)
        rng.shuffle(candidates)
        candidates.sort(key=lambda pid: (need_window[pid], jitter(ratings[pid])), reverse=True)
        while len(lineup) < 5 and candidates:
            best_pid, best_gap = None, 1e9
            for pid in candidates:
                if need_window[pid] <= 0: continue
                if creates_three_streak(pid, p, schedule.lineups): continue
                if violates_pair_rules_if_add(pid, lineup): continue
                trial = lineup + [pid]
                gap = abs(lineup_score(trial) - target_score)
                if gap < best_gap:
                    best_gap, best_pid = gap, pid
            if best_pid is None:
                for pid in candidates:
                    if creates_three_streak(pid, p, schedule.lineups): continue
                    if violates_pair_rules_if_add(pid, lineup): continue
                    best_pid = pid; break
            if best_pid is None:
                best_pid = candidates[0]
            lineup.append(best_pid)
            candidates.remove(best_pid)
            need_window[best_pid] = max(0, need_window[best_pid] - 1)
            total_play[best_pid] += 1
        lineup = enforce_together_if_needed(lineup)
        schedule.lineups[p] = Lineup(p, lineup[:5])

    # Resto de periodos
    def need(pid): return targets_total[pid] - total_play[pid]
    for p in range(constr + 1, periods + 1):
        lineup = []
        candidates = list(player_ids)
        rng.shuffle(candidates)
        candidates.sort(key=lambda pid: (need(pid), jitter(ratings[pid])), reverse=True)
        while len(lineup) < 5 and candidates:
            best_pid, best_gap = None, 1e9
            for pid in candidates:
                if need(pid) <= 0: continue
                if creates_three_streak(pid, p, schedule.lineups): continue
                if violates_pair_rules_if_add(pid, lineup): continue
                trial = lineup + [pid]
                gap = abs(lineup_score(trial) - target_score)
                if gap < best_gap:
                    best_gap, best_pid = gap, pid
            if best_pid is None:
                for pid in candidates:
                    if creates_three_streak(pid, p, schedule.lineups): continue
                    if violates_pair_rules_if_add(pid, lineup): continue
                    best_pid = pid; break
            if best_pid is None:
                best_pid = candidates[0]
            lineup.append(best_pid)
            candidates.remove(best_pid)
            total_play[best_pid] += 1
        lineup = enforce_together_if_needed(lineup)
        schedule.lineups[p] = Lineup(p, lineup[:5])

    # Ajuste fino por equilibrio
    def try_improve_period(p: int):
        players = schedule.lineups[p].players
        current = lineup_score(players)
        tries = 0
        while abs(current - target_score) > tolerance and tries < 40:
            out_idx = random.randrange(0, 5)
            out_pid = players[out_idx]
            outsider = [q for q in player_ids if q not in players]
            if not outsider:
                break
            in_pid = random.choice(outsider)
            if creates_three_streak(in_pid, p, schedule.lineups):
                tries += 1; continue
            trial = players.copy(); trial[out_idx] = in_pid
            hard, _, _ = violates_rules(trial, team)
            if hard:
                tries += 1; continue
            if abs(lineup_score(trial) - target_score) < abs(current - target_score):
                schedule.lineups[p].players = trial
                players = trial
                current = lineup_score(players)
            tries += 1
    for p in range(1, periods + 1):
        try_improve_period(p)

    return schedule

# ===========================
# UI: Auth + Nav
# ===========================
def sidebar_auth():
    st.sidebar.header("Usuario")
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "current_team_id" not in st.session_state:
        st.session_state.current_team_id = None
    if "current_schedule" not in st.session_state:
        st.session_state.current_schedule = None

    if st.session_state.auth_user:
        st.sidebar.success(f"Sesión: {st.session_state.auth_user}")
        if st.sidebar.button("Cerrar sesión"):
            st.session_state.auth_user = None
            st.session_state.current_team_id = None
            st.session_state.current_schedule = None
            st.rerun()
        return

    tab1, tab2 = st.sidebar.tabs(["Iniciar sesión", "Registrarse"])
    with tab1:
        u = st.text_input("Usuario", key="login_u")
        p = st.text_input("Contraseña", type="password", key="login_p")
        if st.button("Entrar"):
            ok, msg = login_user(u, p)
            if ok:
                st.session_state.auth_user = u
                st.success(msg); st.rerun()
            else:
                st.error(msg)
    with tab2:
        u = st.text_input("Usuario nuevo", key="reg_u")
        p = st.text_input("Contraseña", type="password", key="reg_p")
        if st.button("Crear cuenta"):
            ok, msg = register_user(u, p)
            if ok: st.success(msg)
            else: st.error(msg)

def sidebar_nav():
    st.sidebar.header("Navegación")
    return st.sidebar.radio(
        "Ir a:",
        options=[
            "Equipo y Jugadores",
            "Quintetos Manuales",
            "Quintetos Automáticos",
            "Validación",
            "Valoraciones",
            "Reglas",
        ],
    )

# ===========================
# UI: Páginas
# ===========================
def _team_selector_inline(title: str):
    user = st.session_state.auth_user
    teams = db_teams_by_owner(user)
    if not teams:
        return None
    labels = {t["id"]: t["name"] for t in teams}
    ids = list(labels.keys())
    current = st.session_state.current_team_id if st.session_state.current_team_id in ids else ids[0]
    choice = st.selectbox(title, options=ids, index=ids.index(current), format_func=lambda x: labels[x])
    if choice != st.session_state.current_team_id:
        st.session_state.current_team_id = choice
        st.session_state.current_schedule = None
        st.rerun()
    return choice

def page_team_players():
    st.header("Equipo y Jugadores/as")
    user = st.session_state.auth_user
    if not user:
        st.info("Inicia sesión para gestionar equipos.")
        return

    # Selector y creación
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tus equipos")
        teams = db_teams_by_owner(user)
        if teams:
            labels = {t["id"]: t["name"] for t in teams}
            ids = list(labels.keys())
            choice = st.selectbox("Selecciona equipo", options=ids,
                                  index=(ids.index(st.session_state.current_team_id) if st.session_state.current_team_id in ids else 0),
                                  format_func=lambda x: labels[x])
            st.session_state.current_team_id = choice
        else:
            st.warning("No tienes equipos creados.")
    with col2:
        st.subheader("Crear nuevo equipo")
        new_name = st.text_input("Nombre del equipo")
        new_cat = st.selectbox("Categoría", list(CATEGORIES.keys()))
        if st.button("Crear equipo"):
            if not new_name:
                st.error("Introduce un nombre.")
            else:
                t = db_team_create(user, new_name, new_cat)
                st.session_state.current_team_id = t["id"]
                st.success("Equipo creado."); st.rerun()

    if not st.session_state.current_team_id:
        return

    # Edición + Jugadoras
    team = load_team(st.session_state.current_team_id)
    st.markdown("### Equipo seleccionado")
    e1, e2, e3 = st.columns([3,2,1])
    with e1:
        new_name = st.text_input("Nombre", value=team.name, key="team_name_edit")
    with e2:
        new_cat = st.selectbox("Categoría", list(CATEGORIES.keys()),
                               index=list(CATEGORIES.keys()).index(team.category), key="team_cat_edit")
    with e3:
        if st.button("Guardar cambios"):
            team.name = new_name; team.category = new_cat
            save_team_meta(team)
            st.success("Equipo actualizado.")

    with st.expander("Añadir jugador/a"):
        pname = st.text_input("Nombre")
        prating = st.slider("Puntuación (1-5)", 1, 5, 3)
        if st.button("Añadir"):
            if not pname: st.error("Introduce un nombre válido.")
            else:
                db_player_add(team.id, pname, int(prating))
                st.success("Jugador/a añadido."); st.rerun()

    if team.players:
        st.write("Plantilla")
        for pid, p in list(team.players.items()):
            c1, c2, c3 = st.columns([5,3,1])
            with c1:
                new_name = st.text_input("Nombre", value=p.name, key=f"nm_{pid}")
            with c2:
                new_rating = st.slider("Puntuación", 1, 5, int(p.rating), key=f"rtg_{pid}")
            with c3:
                if st.button("Guardar", key=f"sv_{pid}"):
                    db_player_update(pid, new_name, int(new_rating))
                    st.success("Guardado.")
                if st.button("🗑️", key=f"del_{pid}"):
                    db_player_delete(pid)
                    st.warning("Eliminado."); st.rerun()
    else:
        st.info("Añade al menos 5 jugadores/as para poder crear quintetos.")

def page_manual_lineups():
    st.header("Quintetos Manuales")
    if not st.session_state.auth_user:
        st.info("Inicia sesión para continuar."); return
    if not _team_selector_inline("Equipo para crear quintetos manuales"):
        st.info("No tienes equipos todavía."); return
    team = load_team(st.session_state.current_team_id)
    if len(team.players) < 5:
        st.warning("Necesitas al menos 5 jugadores/as."); return

    category = team.category
    periods = CATEGORIES[category]["periods"]
    if st.session_state.current_schedule is None or st.session_state.current_schedule.category != category:
        st.session_state.current_schedule = Schedule(category, periods, CATEGORIES[category]["constrained_periods"])
    sched: Schedule = st.session_state.current_schedule

    player_labels = {pid: f"{team.players[pid].name} (Punt {team.players[pid].rating})" for pid in team.players}
    for p in range(1, periods + 1):
        st.markdown(f"Periodo {p}")
        current = sched.lineups.get(p).players if sched.lineups.get(p) else []
        selected = st.multiselect(
            f"Selecciona 5 jugadores/as para el periodo {p}",
            options=list(team.players.keys()),
            default=current,
            format_func=lambda x: player_labels[x],
            key=f"sel_{p}"
        )
        if len(selected) > 5:
            st.warning("Has seleccionado más de 5. Se tomarán los 5 primeros.")
            selected = selected[:5]
        if len(selected) == 5:
            hard, soft_apart, _ = violates_rules(selected, team)
            if hard:
                st.error("Este quinteto viola reglas personalizadas en modo estricto.")
            elif soft_apart:
                st.warning(" | ".join(soft_apart))
            sched.lineups[p] = Lineup(p, selected)
            score = sum(team.players[pid].rating for pid in selected)
            st.caption(f"Puntuación del quinteto: {score}")
        else:
            if p in sched.lineups:
                del sched.lineups[p]

    if st.button("Validar configuración manual"):
        result = validate_schedule(team, sched)
        if result["errors"]:
            st.error("\n".join(result["errors"]))
        else:
            st.success("Cumple la norma pasarela en número mínimo de juego/descanso.")
        if result["warnings"]:
            st.warning("\n".join(result["warnings"]))

def page_auto_lineups():
    st.header("Quintetos Automáticos")
    if not st.session_state.auth_user:
        st.info("Inicia sesión para continuar."); return
    if not _team_selector_inline("Equipo para crear quintetos automáticos"):
        st.info("No tienes equipos todavía."); return
    team = load_team(st.session_state.current_team_id)
    if len(team.players) < 5:
        st.warning("Necesitas al menos 5 jugadores/as."); return

    seed_best = st.checkbox("Usar quintetos mejor valorados como semilla", value=True)
    variability = st.slider("Variabilidad (soluciones distintas)", 0.0, 1.0, 0.15)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Generar nueva propuesta"):
            try:
                sched = generate_automatic_schedule(team, seed_top_lineups=seed_best, variability=variability)
                st.session_state.current_schedule = sched
                st.success("Propuesta generada.")
            except Exception as e:
                st.error(str(e))
    with colB:
        if st.button("Generar otra diferente"):
            try:
                sched = generate_automatic_schedule(team, seed_top_lineups=seed_best, variability=variability)
                st.session_state.current_schedule = sched
                st.success("Nueva propuesta generada.")
            except Exception as e:
                st.error(str(e))

    if st.session_state.current_schedule and st.session_state.current_schedule.category == team.category:
        sched = st.session_state.current_schedule
        player_labels = {pid: f"{team.players[pid].name} (Punt {team.players[pid].rating})" for pid in team.players}
        st.divider()
        st.subheader("Ajustes rápidos")
        for p in range(1, sched.periods + 1):
            l = sched.lineups.get(p)
            if not l:
                continue
            st.markdown(f"Periodo {p}")
            selected = st.multiselect(
                "Quinteto",
                options=list(team.players.keys()),
                default=l.players,
                format_func=lambda x: player_labels[x],
                key=f"auto_{p}",
            )
            if len(selected) == 5:
                hard, soft_apart, _ = violates_rules(selected, team)
                if hard:
                    st.error("Este quinteto viola reglas personalizadas en modo estricto.")
                elif soft_apart:
                    st.warning(" | ".join(soft_apart))
                sched.lineups[p].players = selected
                score = sum(team.players[pid].rating for pid in selected)
                st.caption(f"Puntuación del quinteto: {score}")
        if st.button("Validar propuesta"):
            result = validate_schedule(team, sched)
            if result["errors"]:
                st.error("\n".join(result["errors"]))
            else:
                st.success("Cumple la norma pasarela en número mínimo de juego/descanso.")
            if result["warnings"]:
                st.warning("\n".join(result["warnings"]))

def page_validation():
    st.header("Validación de Norma Pasarela")
    team_id = st.session_state.current_team_id
    if not team_id:
        st.info("Selecciona o crea un equipo primero."); return
    team = load_team(team_id)
    sched = st.session_state.current_schedule
    if not sched:
        st.info("No hay un calendario de quintetos en memoria. Crea uno manual o automático."); return
    result = validate_schedule(team, sched)
    st.subheader("Resultado")
    if result["errors"]:
        st.error("\n".join(result["errors"]))
    else:
        st.success("Cumple la norma pasarela en los periodos restringidos.")
    if result["warnings"]:
        st.warning("\n".join(result["warnings"]))

    st.subheader("Resumen de quintetos")
    ratings = {pid: team.players[pid].rating for pid in team.players}
    for p in range(1, sched.periods + 1):
        l = sched.lineups.get(p)
        if not l:
            st.write(f"Periodo {p}: (sin definir)")
        else:
            names = ", ".join([team.players[pid].name for pid in l.players])
            score = sum(ratings.get(pid, 3) for pid in l.players)
            st.write(f"Periodo {p}: {names} — Puntuación: {score}")

def page_ratings():
    st.header("Valoraciones de Quintetos (introducción manual)")
    team_id = st.session_state.current_team_id
    if not team_id:
        st.info("Selecciona o crea un equipo primero."); return
    team = load_team(team_id)

    st.subheader("Añadir un quinteto y valorarlo")
    player_labels = {pid: f"{team.players[pid].name} (Punt {team.players[pid].rating})" for pid in team.players}
    picked = st.multiselect(
        "Elige 5 jugadores/as",
        options=list(team.players.keys()),
        format_func=lambda x: player_labels[x],
    )
    note_score = st.slider("Nota del quinteto (0-5)", 0, 5, 3)
    notes = st.text_input("Notas (opcional)")
    if st.button("Guardar quinteto valorado"):
        if len(picked) != 5:
            st.error("Debes seleccionar exactamente 5 jugadores/as.")
        else:
            sig = ",".join(sorted(picked))
            db_rate_lineup(team.id, sig, int(note_score), notes)
            st.success("Quinteto guardado y valorado.")

    st.subheader("Top quintetos guardados")
    top_lineups = db_top_lineups(team.id, top_k=10)
    if not top_lineups:
        st.info("Sin valoraciones aún.")
    else:
        for i, players in enumerate(top_lineups, start=1):
            names = ", ".join([team.players.get(pid, Player(pid, pid, 3)).name for pid in players])
            st.write(f"#{i}: {names}")

def page_rules():
    st.header("Reglas de combinación de jugadoras")
    team_id = st.session_state.current_team_id
    if not team_id:
        st.info("Selecciona o crea un equipo primero."); return
    team = load_team(team_id)

    team.rules.setdefault("together", [])
    team.rules.setdefault("apart", [])
    team.rules.setdefault("strict", False)

    st.subheader("Modo de aplicación")
    strict = st.checkbox("Hacer reglas estrictas", value=bool(team.rules.get("strict", False)))
    if strict != team.rules.get("strict", False):
        db_rules_set_strict(team.id, strict)
        st.success("Modo de reglas actualizado."); st.rerun()

    if not team.players:
        st.info("Añade jugadoras en la página de Equipo."); return

    player_labels = {pid: f"{team.players[pid].name} (Punt {team.players[pid].rating})" for pid in team.players}

    c1, c2 = st.columns(2)

    # --- No juntas (añadir varias) ---
    with c1:
        st.markdown("No juntas")
        apart_a = st.selectbox("Jugadora A", options=list(team.players.keys()),
                               format_func=lambda x: player_labels[x], key="apart_a")
        apart_bs = st.multiselect("Jugadoras B (se crearán varias parejas A-B)",
                                  options=[k for k in team.players.keys() if k != apart_a],
                                  format_func=lambda x: player_labels[x], key="apart_bs")
        if st.button("Añadir pares 'No juntas'"):
            to_add = []
            for b in apart_bs:
                pair = _normalize_pair(apart_a, b)
                if pair and not _has_pair(team.rules["apart"], pair):
                    to_add.append(pair)
            db_rules_add_pairs(team.id, "rules_apart", to_add)
            if to_add:
                st.success(f"Se añadieron {len(to_add)} pareja(s) a 'No juntas'."); st.rerun()
            else:
                st.warning("No se añadió ninguna pareja (duplicadas o no válidas).")
        if team.rules["apart"]:
            st.write("Pares definidos:")
            for x, y in team.rules["apart"]:
                cols = st.columns([6,1])
                with cols[0]:
                    st.write(f"- {player_labels.get(x, x)}  /  {player_labels.get(y, y)}")
                with cols[1]:
                    if st.button("🗑️", key=f"del_apart_{x}_{y}"):
                        db_rules_delete_pair(team.id, "rules_apart", x, y)
                        st.warning("Par eliminado."); st.rerun()

    # --- Sí juntas (añadir varias) ---
    with c2:
        st.markdown("Sí juntas")
        tog_a = st.selectbox("Jugadora A", options=list(team.players.keys()),
                             format_func=lambda x: player_labels[x], key="tog_a")
        tog_bs = st.multiselect("Jugadoras B (se crearán varias parejas A-B)",
                                options=[k for k in team.players.keys() if k != tog_a],
                                format_func=lambda x: player_labels[x], key="tog_bs")
        if st.button("Añadir pares 'Sí juntas'"):
            to_add = []
            for b in tog_bs:
                pair = _normalize_pair(tog_a, b)
                if pair and not _has_pair(team.rules["together"], pair):
                    to_add.append(pair)
            db_rules_add_pairs(team.id, "rules_together", to_add)
            if to_add:
                st.success(f"Se añadieron {len(to_add)} pareja(s) a 'Sí juntas'."); st.rerun()
            else:
                st.warning("No se añadió ninguna pareja (duplicadas o no válidas).")
        if team.rules["together"]:
            st.write("Pares definidos:")
            for x, y in team.rules["together"]:
                cols = st.columns([6,1])
                with cols[0]:
                    st.write(f"- {player_labels.get(x, x)}  /  {player_labels.get(y, y)}")
                with cols[1]:
                    if st.button("🗑️", key=f"del_together_{x}_{y}"):
                        db_rules_delete_pair(team.id, "rules_together", x, y)
                        st.warning("Par eliminado."); st.rerun()

    st.info("Las reglas se aplican en la validación y en la generación automática. En modo no estricto, solo se avisa de 'no juntas'.")

# ===========================
# Main
# ===========================
sidebar_auth()
if st.session_state.auth_user:
    page = sidebar_nav()
    if page == "Equipo y Jugadores":
        page_team_players()
    elif page == "Quintetos Manuales":
        page_manual_lineups()
    elif page == "Quintetos Automáticos":
        page_auto_lineups()
    elif page == "Validación":
        page_validation()
    elif page == "Valoraciones":
        page_ratings()
    elif page == "Reglas":
        page_rules()
else:
    st.title("Quintetos con Norma Pasarela (FBCV)")
    st.markdown(
        """
        Inicia sesión desde la barra lateral para:
        - Crear equipos (con categoría) e introducir jugadoras con puntuación 1-5
        - Diseñar quintetos manuales y automáticos
        - Cumplir la norma pasarela (ventana restringida por categoría y sin 3 seguidos en P1–P6)
        - Definir Reglas: pares sí juntas / no juntas (modo estricto opcional)
        - Guardar y reutilizar tus quintetos valorados manualmente
        """
    )
