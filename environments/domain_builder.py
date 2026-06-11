import itertools
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml
from negmas import LinearAdditiveUtilityFunction, make_issue
from utils.utility import compute_utility
from pathlib import Path
from negmas.inout import Scenario

@dataclass
class DomainSpec:
    domain_name: str
    issues: list
    issue_names: List[str]
    value_lists: Dict[str, List[Any]]
    values: Dict[str, Dict[Any, float]]
    weights: Dict[str, float]
    reserved_values: Dict[str, float]
    action_library: List[Dict[str, Any]]
    utility_min: float
    utility_max: float
    best_outcome: Dict[str, Any]
    worst_outcome: Dict[str, Any]
    utility_min_opponent: float = None
    utility_max_opponent: float = None
    best_outcome_opponent: Dict[str, Any] = None
    worst_outcome_opponent: Dict[str, Any] = None
    values_opponent: Dict[str, Dict[Any, float]] = None
    weights_opponent: Dict[str, float] = None

    


def _normalize_value_types(issue_defs, values_dict):
    if values_dict is None:
        values_dict = {}
    for issue in issue_defs:
        name = issue["name"]
        ref_type = type(issue["values"][0]) if issue.get("values") else str
        issue_map = values_dict.get(name, {}) or {}

        new_map = {}
        for k, v in issue_map.items():
            try:
                new_k = ref_type(k)
            except Exception:
                new_k = k
            new_map[new_k] = float(v)

        values_dict[name] = new_map
    return values_dict


def _load_yaml_domain(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_genius_like_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    issues, values, weights = [], {}, {}

    for issue_node in root.iter("issue"):
        name = issue_node.attrib["name"]

        issue_values = []
        value_map = {}

        for item in issue_node:
            raw = item.attrib.get("value") or item.attrib.get("name") or item.text
            raw = str(raw).strip()

            try:
                casted = int(raw)
            except Exception:
                try:
                    casted = float(raw)
                except Exception:
                    casted = raw

            util = float(item.attrib.get("utility", 0.0))

            issue_values.append(casted)
            value_map[casted] = util

        issues.append({"name": name, "values": issue_values})
        values[name] = value_map
        weights[name] = float(issue_node.attrib.get("weight", 1.0))

    return {
        "domain_name": root.attrib.get("name", "domain"),
        "issues": issues,
        "values": values,
        "weights": weights,
    }


def _enumerate_action_library(issue_defs, max_combinations=256):
    names = [x["name"] for x in issue_defs]
    values = [x["values"] for x in issue_defs]

    combos = list(itertools.product(*values))
    if len(combos) > max_combinations:
        combos = combos[:max_combinations]

    return [{names[i]: combo[i] for i in range(len(names))} for combo in combos]


def _utility_range(ufun, action_library, issue_names):
    utils = [compute_utility(ufun, o, issue_names) for o in action_library]
    min_idx = min(range(len(utils)), key=lambda i: utils[i])
    max_idx = max(range(len(utils)), key=lambda i: utils[i])
    return utils[min_idx], utils[max_idx], action_library[max_idx], action_library[min_idx]


def fix_issue_indexing(xml_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    idx = 0
    for issue in root.iter("issue"):
        issue.set("index", str(idx))
        idx += 1

    idx = 0
    for w in root.iter("weight"):
        w.set("index", str(idx))
        idx += 1

    tree.write(xml_path)


import xml.etree.ElementTree as ET
from pathlib import Path
from negmas import LinearAdditiveUtilityFunction, make_issue

def parse_utility_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    issues = []
    values = {}
    weights = {}

    # parse issues
    for issue in root.iter("issue"):
        name = issue.attrib["name"]
        vals = []
        val_map = {}

        for item in issue.iter("item"):
            v = item.attrib["value"]
            eval_ = float(item.attrib.get("evaluation", 0))
            vals.append(v)
            val_map[v] = eval_

        issues.append({"name": name, "values": vals})
        values[name] = val_map

    # parse weights
    for w in root.iter("weight"):
        weights[int(w.attrib["index"])] = float(w.attrib["value"])

    # mapping index → issue name
    weights_named = {}
    for i, issue in enumerate(issues):
        weights_named[issue["name"]] = weights.get(i, 1.0)

    return issues, values, weights_named


def build_domain(domain_path: str, max_combinations: int = 256):

    path = Path(domain_path)
    # 1) Folder GENIUS/ANAC (INI MASIH HARD CODED UNTUK DOMAIN LAPTOP SAJA DARI NEGMAS (ANAC)). LEARNER: BUYER, OPPONENT: SELLER
    if path.is_dir():
        buyer_file = list(path.glob("*buyer*.xml"))[0]
        seller_file = list(path.glob("*seller*.xml"))[0]

        issue_defs, values_user, weights_user = parse_utility_xml(buyer_file)
        _, values_opp, weights_opp = parse_utility_xml(seller_file)

        issue_objs = [make_issue(values=x["values"], name=x["name"]) for x in issue_defs]
        issue_names = [x["name"] for x in issue_defs]

        learner_ufun = LinearAdditiveUtilityFunction(
            issues=issue_objs,
            values=values_user,
            weights=weights_user,
        )

        opponent_ufun = LinearAdditiveUtilityFunction(
            issues=issue_objs,
            values=values_opp,
            weights=weights_opp,
        )

        action_library = _enumerate_action_library(issue_defs, max_combinations)

        umin, umax, best, worst = _utility_range(learner_ufun, action_library, issue_names)
        umin_o, umax_o, best_o, worst_o = _utility_range(opponent_ufun, action_library, issue_names)

        return DomainSpec(
            domain_name=path.name,
            issues=issue_objs,
            issue_names=issue_names,
            value_lists={x["name"]: x["values"] for x in issue_defs},
            values=values_user,
            values_opponent=values_opp,
            weights=weights_user,
            weights_opponent=weights_opp,
            reserved_values={"learner": 0.0, "opponent": 0.0},
            action_library=action_library,
            utility_min=umin,
            utility_max=umax,
            best_outcome=best,
            worst_outcome=worst,
            utility_min_opponent=umin_o,
            utility_max_opponent=umax_o,
            best_outcome_opponent=best_o,
            worst_outcome_opponent=worst_o,
        ), learner_ufun, opponent_ufun

    ext = os.path.splitext(domain_path)[1].lower()

    if ext in [".yaml", ".yml"]:
        cfg = _load_yaml_domain(domain_path)
    elif ext == ".xml":
        cfg = _load_genius_like_xml(domain_path)
    else:
        raise ValueError(f"Format domain tidak didukung: {domain_path}")

    user_state_representation = cfg.get("user", {}).get("state_representation", None)
    opponent_state_representation = cfg.get("opponent", {}).get("state_representation", None)
    # CFG USER
    cfg_user = cfg.get("user", {})
    if cfg_user:
        issue_defs = cfg_user["issues"]

        issue_objs = [make_issue(values=x["values"], name=x["name"]) for x in issue_defs]
        issue_names = [x["name"] for x in issue_defs]

        values_dict = _normalize_value_types(issue_defs, cfg_user.get("values", {}))
        weights_dict = cfg_user.get("weights", {}) or {name: 1.0 for name in issue_names}

        if len(weights_dict) == 0:
            weights_dict = {name: 1.0 for name in issue_names}

        s = sum(weights_dict.values())
        if s <= 0:
            raise ValueError("Total weight must be positive")
        weights_dict = {k: float(v) / s for k, v in weights_dict.items()}

        # print("\n[FINAL VALUES STRUCTURE]")
        # for k, v in values_dict.items():
        #     # print(k, "->", list(v.keys()))

        ufun = LinearAdditiveUtilityFunction(
            issues=issue_objs,
            values=values_dict,
            weights=weights_dict,
        )

        action_library = _enumerate_action_library(issue_defs, max_combinations)
        umin, umax, best, worst = _utility_range(ufun, action_library, issue_names)

        if umax - umin < 1e-8:
            umax = umin + 1e-6
    
    else:
        raise ValueError("Bagian 'user' harus didefinisikan dalam konfigurasi domain")


    # CFG OPPONENT
    cfg_opponent = cfg.get("opponent", {})
    if cfg_opponent:
        issue_defs_opponent = cfg_opponent["issues"]
        values_dict_opponent = _normalize_value_types(issue_defs_opponent, cfg_opponent.get("values", {}))
        weights_dict_opponent = cfg_opponent.get("weights", {}) or {name: 1.0 for name in issue_names}

        s_opponent = sum(weights_dict_opponent.values())
        if s_opponent <= 0:
            raise ValueError("Total weight for opponent must be positive")
        weights_dict_opponent = {k: float(v) / s_opponent for k, v in weights_dict_opponent.items()}

        ufun_opponent = LinearAdditiveUtilityFunction(
            issues=[make_issue(values=x["values"], name=x["name"]) for x in issue_defs_opponent],
            values=values_dict_opponent,
            weights=weights_dict_opponent,
        )

        umin_opponent, umax_opponent, best_opponent, worst_opponent = _utility_range(ufun_opponent, action_library, issue_names)

        if umax_opponent - umin_opponent < 1e-8:
            umax_opponent = umin_opponent + 1e-6
    else:
        raise ValueError("Bagian 'opponent' harus didefinisikan dalam konfigurasi domain")


    return DomainSpec(
        domain_name=cfg_user.get("domain_name", os.path.basename(domain_path)),
        issues=issue_objs,
        issue_names=issue_names,
        value_lists={x["name"]: x["values"] for x in issue_defs},
        values=values_dict,
        values_opponent=values_dict_opponent if cfg_opponent else None,
        weights=weights_dict,
        weights_opponent=weights_dict_opponent if cfg_opponent else None,
        reserved_values=cfg.get("reserved_values", {"learner": 0.0, "opponent": 0.0}),
        action_library=action_library,
        utility_min=umin,
        utility_min_opponent=umin_opponent if cfg_opponent else None,
        utility_max=umax,
        utility_max_opponent=umax_opponent if cfg_opponent else None,
        best_outcome=best,
        best_outcome_opponent=best_opponent if cfg_opponent else None,
        worst_outcome=worst,
        worst_outcome_opponent=worst_opponent if cfg_opponent else None,
    ), ufun, ufun_opponent if cfg_opponent else None