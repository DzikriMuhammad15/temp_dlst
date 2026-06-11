from collections.abc import Mapping


def _unwrap_offer(offer):
    if offer is None:
        return None

    # SAOState / negotiation state => use agreement first, then current_offer
    if hasattr(offer, "agreement"):
        agreement = getattr(offer, "agreement")
        if agreement is not None:
            offer = agreement

    if hasattr(offer, "current_offer"):
        current_offer = getattr(offer, "current_offer")
        if current_offer is not None:
            offer = current_offer

    return offer


def compute_utility(ufun, offer, issue_names):
    """
    Safe utility evaluation.

    Priority:
    1) unwrap SAOState-like objects into agreement/current_offer
    2) if mapping, try tuple order using issue_names first (more stable)
    3) fallback to direct NegMAS evaluation
    """
    offer = _unwrap_offer(offer)

    if offer is None:
        return 0.0

    if isinstance(offer, Mapping) and issue_names:
        try:
            return float(ufun(tuple(offer[k] for k in issue_names)))
        except Exception:
            pass

    try:
        return float(ufun(offer))
    except Exception:
        try:
            if isinstance(offer, Mapping) and issue_names:
                return float(ufun(tuple(offer[k] for k in issue_names)))
            raise
        except Exception as e:
            # print("\n[UTILITY ERROR]")
            # print("Offer:", offer)
            # print("Issue names:", issue_names)
            if isinstance(offer, Mapping) and issue_names:
                try:
                    print("Tuple:", tuple(offer[k] for k in issue_names))
                except Exception:
                    pass
            raise e


def object_to_dict(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    if isinstance(obj, dict):
        return {
            str(k): object_to_dict(v)
            for k, v in obj.items()
        }

    # FIX DI SINI
    if isinstance(obj, (list, tuple, set)):
        return [
            object_to_dict(v)
            for v in obj
        ]

    result = {}

    for attr in dir(obj):
        if attr.startswith("_"):
            continue

        try:
            value = getattr(obj, attr)

            if callable(value):
                continue

            result[attr] = object_to_dict(value)

        except Exception:
            result[attr] = "<unreadable>"

    return result