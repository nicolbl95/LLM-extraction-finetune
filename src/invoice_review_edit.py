"""Session-local human edits kept separate from extraction and P1 review data."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import hmac
import json
import math
import os
import re
import secrets
from uuid import uuid4

SCALAR_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")
LINE_FIELDS = ("description", "quantity", "unit_price", "total_ht")
NUMERIC_FIELDS = {"quantity", "unit_price", "total_ht"}
STATE_VERSION = 1
TOLERANCE = Decimal("0.02")
EDIT_UI_ENV = "INVOICE_REVIEW_EDIT"
_CONFIRMATION_KEY = secrets.token_bytes(32)


def edit_ui_enabled(environ=None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(EDIT_UI_ENV, "")).strip().casefold() in {
        "1", "true", "yes", "on",
    }


class EditStateError(ValueError):
    """The submitted edit state or transition is invalid."""


def _copy(value):
    return deepcopy(value)


def _confirmation_token(state: dict, field_id: str) -> str:
    payload = json.dumps(
        {
            "field": field_id,
            "value": state["draft"][field_id],
            "origin": state["draft_origins"][field_id],
            "decision": state["decisions"][field_id],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hmac.new(_CONFIRMATION_KEY, payload, hashlib.sha256).hexdigest()


def _assessment_for(review: dict, original_index: int | None, field: str):
    assessments = review.get("field_assessments") or {}
    if original_index is None:
        return _copy(assessments.get(field) or {})
    return _copy(
        assessments.get(f"line_items[{original_index}].{field}")
        or assessments.get(f"line_items.{original_index}.{field}")
        or {}
    )


def _field_key(row_id: str | None, field: str) -> str:
    return field if row_id is None else f"row:{row_id}:{field}"


def _get_path_value(payload: dict, original_index: int | None, field: str):
    if original_index is None:
        return _copy(payload.get(field))
    rows = payload.get("line_items") or []
    return _copy(rows[original_index].get(field)) if original_index < len(rows) else None


def initialize_edit_state(strict: dict, review: dict, raw_model: dict | None = None) -> dict:
    """Create independent snapshots and stable row identities for one Gradio session."""
    if not isinstance(strict, dict) or not isinstance(review, dict):
        raise EditStateError("Les résultats stricts et P1 doivent être des objets.")
    strict_copy, review_copy = _copy(strict), _copy(review)
    if raw_model is not None and not isinstance(raw_model, dict):
        raise EditStateError("La lecture brute du modèle doit être un objet.")
    model_copy = _copy(raw_model or {})
    rows = strict_copy.get("line_items")
    model_rows = model_copy.get("line_items")
    review_rows = review_copy.get("line_items")
    rows = [] if rows is None else rows
    model_rows = [] if model_rows is None else model_rows
    review_rows = [] if review_rows is None else review_rows
    if not all(isinstance(value, list) for value in (rows, model_rows, review_rows)):
        raise EditStateError("Les lignes de facture doivent être des listes.")
    if any(row is not None and not isinstance(row, dict)
           for collection in (rows, model_rows, review_rows) for row in collection):
        raise EditStateError("Chaque ligne de facture doit être un objet ou une valeur absente.")
    if not isinstance(review_copy.get("field_assessments", {}), dict):
        raise EditStateError("Les éléments de preuve P1 sont incohérents.")
    row_count = max(len(rows), len(model_rows), len(review_rows))
    row_ids = [str(uuid4()) for _ in range(row_count)]

    fields: dict[str, dict] = {}
    draft: dict[str, object] = {}
    decisions: dict[str, str] = {}
    draft_origins: dict[str, str] = {}
    confirmations: dict[str, bool] = {}
    confirmation_tokens: dict[str, str | None] = {}
    for field in SCALAR_FIELDS:
        fields[field] = {
            "base": _get_path_value(strict_copy, None, field),
            "proposal": _copy(model_copy.get(field)),
            "assessment": _assessment_for(review_copy, None, field),
            "origin": "strict",
        }
        draft[field] = _copy(fields[field]["base"])
        decisions[field] = "pending" if fields[field]["proposal"] is not None else "none"
        draft_origins[field] = "strict"
        confirmations[field] = False
        confirmation_tokens[field] = None

    row_metadata: dict[str, dict] = {}
    row_order: list[str] = []
    for index, row_id in enumerate(row_ids):
        row_order.append(row_id)
        base_row = rows[index] if index < len(rows) and isinstance(rows[index], dict) else {}
        model_row = model_rows[index] if index < len(model_rows) and isinstance(model_rows[index], dict) else {}
        row_metadata[row_id] = {"original_index": index if index < len(rows) else None,
                                "origin": "extracted" if index < len(rows) else "model"}
        for field in LINE_FIELDS:
            key = _field_key(row_id, field)
            fields[key] = {
                "base": _copy(base_row.get(field)),
                "proposal": _copy(model_row.get(field)),
                "assessment": _assessment_for(
                    review_copy, index if index < len(rows) or index < len(review_rows) else None,
                    field,
                ),
                "origin": (
                    "strict" if row_metadata[row_id]["origin"] == "extracted" else "model"
                ),
            }
            draft[key] = _copy(fields[key]["base"])
            decisions[key] = "pending" if fields[key]["proposal"] is not None else "none"
            draft_origins[key] = (
                "strict" if row_metadata[row_id]["origin"] == "extracted" else "model"
            )
            confirmations[key] = False
            confirmation_tokens[key] = None
    return {
        "version": STATE_VERSION,
        "original_strict": strict_copy,
        "original_review": review_copy,
        "original_model": model_copy,
        "fields": fields,
        "draft": draft,
        "decisions": decisions,
        "draft_origins": draft_origins,
        "confirmed": confirmations,
        "confirmation_tokens": confirmation_tokens,
        "rows": row_metadata,
        "row_order": row_order,
        "validation_errors": {},
    }


def _validate_state(state: dict) -> None:
    if (not isinstance(state, dict) or state.get("version") != STATE_VERSION
            or not all(isinstance(state.get(key), dict) for key in
                       ("original_strict", "original_review", "original_model",
                        "fields", "draft", "decisions", "draft_origins",
                        "confirmed", "confirmation_tokens", "rows"))
            or not isinstance(state.get("row_order"), list)):
        raise EditStateError("État de révision invalide ou expiré.")
    field_keys = set(state["fields"])
    if any(not isinstance(key, str) or not isinstance(value, dict)
           for key, value in state["fields"].items()):
        raise EditStateError("Métadonnées de champ invalides.")
    if any(not isinstance(row_id, str) or not isinstance(metadata, dict)
           for row_id, metadata in state["rows"].items()):
        raise EditStateError("Métadonnées de ligne invalides.")
    if any(not isinstance(row_id, str) for row_id in state["row_order"]):
        raise EditStateError("Identifiant de ligne invalide.")
    if any(not isinstance(state[key], dict) or set(state[key]) != field_keys
           for key in (
               "draft", "decisions", "draft_origins", "confirmed",
               "confirmation_tokens",
           )):
        raise EditStateError("Les champs de l'état de révision sont incohérents.")
    if any(not isinstance(confirmed, bool) for confirmed in state["confirmed"].values()):
        raise EditStateError("État de confirmation invalide.")
    for field_id, confirmed in state["confirmed"].items():
        token = state["confirmation_tokens"][field_id]
        if confirmed:
            if not isinstance(token, str) or not hmac.compare_digest(
                token, _confirmation_token(state, field_id),
            ):
                raise EditStateError("La confirmation du champ n'est pas valide.")
        elif token is not None:
            raise EditStateError("Jeton de confirmation inattendu.")
    if any(not isinstance(decision, str)
           or decision not in {"none", "pending", "accepted", "rejected"}
           for decision in state["decisions"].values()):
        raise EditStateError("Décision de proposition invalide.")
    if any(not isinstance(origin, str)
           or origin not in {"strict", "model", "human", "model_proposal", "human_correction"}
           for origin in state["draft_origins"].values()):
        raise EditStateError("Origine de brouillon invalide.")
    if (len(state["row_order"]) != len(set(state["row_order"]))
            or set(state["row_order"]) != set(state["rows"])):
        raise EditStateError("Identifiant de ligne invalide.")
    expected_fields: set[str] = set(SCALAR_FIELDS)
    for row_id in state["row_order"]:
        expected_fields.update(_field_key(row_id, field) for field in LINE_FIELDS)
    if field_keys != expected_fields:
        raise EditStateError("Identifiants de champ incohérents.")
    for metadata in state["fields"].values():
        if (not {"base", "proposal", "assessment", "origin"} <= set(metadata)
                or not isinstance(metadata["assessment"], dict)
                or not isinstance(metadata["origin"], str)
                or metadata["origin"] not in {"strict", "model", "human"}):
            raise EditStateError("Métadonnées de champ invalides.")


def field_choices(state: dict) -> list[tuple[str, str]]:
    _validate_state(state)
    labels = {
        "supplier_name": "Fournisseur",
        "invoice_date": "Date de facture",
        "total_ht": "Total HT",
        "total_ttc": "Total TTC",
        "description": "Description",
        "quantity": "Quantité",
        "unit_price": "Prix unitaire HT",
    }
    choices = [(labels[field], field) for field in SCALAR_FIELDS]
    for position, row_id in enumerate(state["row_order"], 1):
        for field in LINE_FIELDS:
            label = "Total HT de ligne" if field == "total_ht" else labels[field]
            choices.append((f"Ligne {position} — {label}", _field_key(row_id, field)))
    return choices


def _resolve(state: dict, field_id: str):
    _validate_state(state)
    if not isinstance(field_id, str) or field_id not in state["fields"]:
        raise EditStateError("Champ invalide ou absent de cette facture.")
    if field_id.startswith("row:"):
        _, row_id, field = field_id.split(":", 2)
        if row_id not in state["rows"] or field not in LINE_FIELDS:
            raise EditStateError("Identifiant de ligne ou champ invalide.")
        return row_id, field
    if field_id not in SCALAR_FIELDS:
        raise EditStateError("Champ invalide.")
    return None, field_id


def field_details(state: dict, field_id: str) -> dict:
    _resolve(state, field_id)
    metadata = state["fields"][field_id]
    return {
        "original": _copy(metadata["base"]),
        "proposal": _copy(metadata["proposal"]),
        "draft": _copy(state["draft"][field_id]),
        "decision": state["decisions"][field_id],
        "confirmed": bool(state["confirmed"].get(field_id, False)),
        "assessment": _copy(metadata["assessment"]),
    }


def _parse_value(field: str, value):
    if value is None or value == "":
        return None
    if field in NUMERIC_FIELDS or field in {"total_ht", "total_ttc"}:
        if isinstance(value, bool):
            raise EditStateError("Une valeur numérique ne peut pas être booléenne.")
        raw = str(value).strip().replace("\u00a0", "").replace(" ", "")
        if not raw:
            return None
        if not re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)", raw):
            raise EditStateError("Saisissez un nombre sans séparateur de milliers ambigu.")
        if "," in raw and "." in raw:
            raise EditStateError("Utilisez un seul séparateur décimal.")
        try:
            number = Decimal(raw.replace(",", "."))
        except InvalidOperation as exc:
            raise EditStateError("Nombre invalide.") from exc
        if not number.is_finite():
            raise EditStateError("Le nombre doit être fini.")
        try:
            numeric = float(number)
        except (OverflowError, ValueError) as exc:
            raise EditStateError("Le nombre dépasse la précision prise en charge.") from exc
        if not math.isfinite(numeric) or (number != 0 and numeric == 0):
            raise EditStateError("Le nombre dépasse la précision prise en charge.")
        if Decimal(str(numeric)) != number:
            raise EditStateError("Le nombre dépasse la précision prise en charge.")
        return int(number) if number == number.to_integral_value() else float(number)
    if field == "invoice_date":
        if not isinstance(value, str):
            raise EditStateError("La date doit être fournie comme texte.")
        raw_date = str(value).strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
            raise EditStateError("Saisissez une date valide au format AAAA-MM-JJ.")
        try:
            return date.fromisoformat(raw_date).isoformat()
        except ValueError as exc:
            raise EditStateError("Saisissez une date valide au format AAAA-MM-JJ.") from exc
    if field in {"supplier_name", "description"}:
        if not isinstance(value, str):
            raise EditStateError("Le texte doit être fourni comme chaîne de caractères.")
        text = str(value).strip()
        if not text:
            raise EditStateError("Ce champ obligatoire ne peut pas être vide.")
        return text
    raise EditStateError("Champ non modifiable.")


def _decimal(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def validate_draft(state: dict) -> dict:
    """Validate available values without calculating or replacing any financial value."""
    _validate_state(state)
    errors = {}
    incomplete = []
    for field_id in state["fields"]:
        field = field_id.rsplit(":", 1)[-1] if field_id.startswith("row:") else field_id
        value = state["draft"][field_id]
        try:
            parsed = _parse_value(field, value)
            if parsed is None:
                incomplete.append(field_id)
        except EditStateError as exc:
            errors[field_id] = str(exc)

    def numeric_value(field_id: str, field: str):
        try:
            return _decimal(_parse_value(field, state["draft"][field_id]))
        except EditStateError:
            return None

    for row_id in state["row_order"]:
        keys = {field: _field_key(row_id, field) for field in LINE_FIELDS}
        values = {
            field: numeric_value(key, field) for field, key in keys.items()
        }
        quantity = values["quantity"]
        unit_price = values["unit_price"]
        total = values["total_ht"]
        if quantity is not None and unit_price is not None and total is not None:
            try:
                expected = (quantity * unit_price).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            except InvalidOperation:
                message = "Valeurs numériques trop grandes pour vérifier l'arithmétique."
                for field in ("quantity", "unit_price", "total_ht"):
                    errors[keys[field]] = message
                continue
            if abs(expected - total) > TOLERANCE:
                message = "Quantité × prix unitaire ne correspond pas au total HT de la ligne."
                for field in ("quantity", "unit_price", "total_ht"):
                    errors[keys[field]] = message

    line_total_ids = [
        _field_key(row_id, "total_ht") for row_id in state["row_order"]
    ]
    subtotal = numeric_value("total_ht", "total_ht")
    line_totals: list[Decimal | None] = [
        numeric_value(key, "total_ht") for key in line_total_ids
    ]
    if subtotal is not None and line_totals and all(value is not None for value in line_totals):
        actual = sum(
            (value for value in line_totals if isinstance(value, Decimal)),
            Decimal(0),
        )
        if abs(actual - subtotal) > TOLERANCE:
            errors["total_ht"] = "La somme des lignes ne correspond pas au sous-total HT."
            for key in line_total_ids:
                errors[key] = "La somme des lignes ne correspond pas au sous-total HT."
    return {
        "errors": errors,
        "incomplete": incomplete,
        "complete": not errors and not incomplete,
    }


def _affected_confirmations(state: dict, row_id: str | None, field: str) -> set[str]:
    affected = set()
    if row_id is None:
        affected.add(field)
        if field == "total_ht":
            affected.update(_field_key(item, "total_ht") for item in state["row_order"])
    else:
        own = _field_key(row_id, field)
        affected.add(own)
        if field in {"quantity", "unit_price", "total_ht"}:
            affected.add(_field_key(row_id, "total_ht"))
            affected.add("total_ht")
    return affected


def _apply(state: dict, field_id: str, action: str, value=None) -> dict:
    _validate_state(state)
    row_id, field = _resolve(state, field_id)
    if action not in {"accept", "reject", "edit", "confirm"}:
        raise EditStateError("Action de révision invalide.")
    next_state = _copy(state)
    metadata = next_state["fields"][field_id]
    if action == "accept":
        if metadata["proposal"] is None or next_state["decisions"][field_id] == "rejected":
            raise EditStateError("Aucune proposition acceptable pour ce champ.")
        parsed = _parse_value(field, metadata["proposal"])
        if parsed is None:
            raise EditStateError("La proposition ne contient aucune valeur exploitable.")
        next_state["draft"][field_id] = parsed
        next_state["decisions"][field_id] = "accepted"
        next_state["draft_origins"][field_id] = "model_proposal"
    elif action == "reject":
        if metadata["proposal"] is None:
            raise EditStateError("Aucune proposition à rejeter.")
        next_state["decisions"][field_id] = "rejected"
        next_state["draft"][field_id] = _copy(metadata["base"])
        next_state["draft_origins"][field_id] = "strict"
    elif action == "edit":
        next_state["draft"][field_id] = _parse_value(field, value)
        next_state["draft_origins"][field_id] = "human_correction"
    else:
        if (next_state["decisions"][field_id] == "rejected"
                and next_state["draft_origins"][field_id] != "human_correction"):
            raise EditStateError("Une proposition rejetée ne peut pas être confirmée.")
        current = next_state["draft"][field_id]
        if current is None:
            raise EditStateError("Un champ vide ne peut pas être confirmé.")
        _parse_value(field, current)
        validation = validate_draft(next_state)
        if field_id in validation["errors"]:
            raise EditStateError(validation["errors"][field_id])
        next_state["confirmed"][field_id] = True
        next_state["confirmation_tokens"][field_id] = _confirmation_token(
            next_state, field_id,
        )
        return next_state

    for affected in _affected_confirmations(next_state, row_id, field):
        next_state["confirmed"][affected] = False
        next_state["confirmation_tokens"][affected] = None
    next_state["validation_errors"] = validate_draft(next_state)["errors"]
    return next_state


def accept_proposal(state: dict, field_id: str) -> dict:
    return _apply(state, field_id, "accept")


def reject_proposal(state: dict, field_id: str) -> dict:
    return _apply(state, field_id, "reject")


def apply_correction(state: dict, field_id: str, value) -> dict:
    return _apply(state, field_id, "edit", value)


def confirm_field(state: dict, field_id: str) -> dict:
    if isinstance(state, dict) and state.get("document_verified") is False:
        raise EditStateError("Rattachez le document source correspondant avant toute confirmation.")
    return _apply(state, field_id, "confirm")


def restore_edit_state(strict, review, model, anchors, rows, row_order, values, origins, decisions):
    """Rebuild evidence metadata from originals; never accept persisted field metadata/tokens."""
    from uuid import UUID

    state = initialize_edit_state(strict, review, model)
    if not isinstance(anchors, list) or len(anchors) != len(state["row_order"]):
        raise EditStateError("Identités des lignes extraites incohérentes.")
    if (not isinstance(rows, dict) or not isinstance(row_order, list)
            or len(row_order) != len(set(row_order)) or set(row_order) != set(rows)
            or not set(anchors) <= set(rows)):
        raise EditStateError("Ordre des lignes sauvegardées invalide.")
    for row_id in row_order:
        try:
            valid = isinstance(row_id, str) and str(UUID(row_id)) == row_id
        except ValueError:
            valid = False
        if not valid:
            raise EditStateError("Identité de ligne sauvegardée invalide.")
    rebuilt = {key: state["fields"][key] for key in SCALAR_FIELDS}
    for index, row_id in enumerate(anchors):
        original_id = state["row_order"][index]
        if rows[row_id] != state["rows"][original_id]:
            raise EditStateError("Association entre ligne et preuve d'origine invalide.")
        for field in LINE_FIELDS:
            rebuilt[_field_key(row_id, field)] = state["fields"][_field_key(original_id, field)]
    for row_id in row_order:
        if row_id in anchors:
            continue
        if rows[row_id] != {"original_index": None, "origin": "human"}:
            raise EditStateError("Une ligne ajoutée ne peut pas inventer de preuve OCR.")
        for field in LINE_FIELDS:
            rebuilt[_field_key(row_id, field)] = {
                "base": None, "proposal": None, "assessment": {}, "origin": "human",
            }
    state.update(
        fields=rebuilt, rows=_copy(rows), row_order=_copy(row_order),
        draft=_copy(values), draft_origins=_copy(origins), decisions=_copy(decisions),
        confirmed={key: False for key in rebuilt},
        confirmation_tokens={key: None for key in rebuilt},
    )
    validation = validate_draft(state)
    # Arithmetic inconsistencies are valid drafts; malformed field values are not.
    for key, value in state["draft"].items():
        field = key.rsplit(":", 1)[-1]
        _parse_value(field, value)
        decision, origin = state["decisions"][key], state["draft_origins"][key]
        proposal = state["fields"][key]["proposal"]
        if decision in {"accepted", "rejected", "pending"} and proposal is None:
            raise EditStateError("Décision sans proposition d'origine.")
        if origin == "model_proposal" and (
            decision != "accepted" or value != _parse_value(field, proposal)
        ):
            raise EditStateError("Valeur de proposition incohérente.")
        if origin == "strict" and value != state["fields"][key]["base"]:
            raise EditStateError("Valeur d'origine altérée.")
    state["validation_errors"] = validation["errors"]
    return state


def add_line_item(state: dict) -> dict:
    _validate_state(state)
    next_state = _copy(state)
    row_id = str(uuid4())
    next_state["rows"][row_id] = {"original_index": None, "origin": "human"}
    next_state["row_order"].append(row_id)
    next_state["confirmation_tokens"]["total_ht"] = None
    for field in LINE_FIELDS:
        key = _field_key(row_id, field)
        next_state["fields"][key] = {
            "base": None, "proposal": None, "assessment": {}, "origin": "human",
        }
        next_state["draft"][key] = None
        next_state["decisions"][key] = "none"
        next_state["draft_origins"][key] = "human"
        next_state["confirmed"][key] = False
        next_state["confirmation_tokens"][key] = None
    next_state["confirmed"]["total_ht"] = False
    next_state["validation_errors"] = validate_draft(next_state)["errors"]
    return next_state


def confirmed_view(state: dict) -> dict:
    """Build a separate, partial human-confirmed result; never imply accounting export."""
    validation = validate_draft(state)
    fields = {}
    for field_id, confirmed in state["confirmed"].items():
        if confirmed and state["draft"].get(field_id) is not None:
            fields[field_id] = {
                "value": _copy(state["draft"][field_id]),
                "status": "human_confirmed",
                "proposal_decision": state["decisions"].get(field_id),
                "value_origin": state["draft_origins"].get(field_id),
            }
    line_items = []
    for row_id in state["row_order"]:
        row_fields = {
            field: fields[_field_key(row_id, field)]
            for field in LINE_FIELDS if _field_key(row_id, field) in fields
        }
        if row_fields:
            line_items.append({
                "row_id": row_id,
                "origin": state["rows"][row_id]["origin"],
                "fields": row_fields,
            })
    required: list[str] = list(SCALAR_FIELDS)
    for row_id in state["row_order"]:
        required.extend(_field_key(row_id, field) for field in LINE_FIELDS)
    all_confirmed = all(
        state["confirmed"].get(field_id)
        and state["draft"].get(field_id) is not None
        for field_id in required
    )
    return {
        "status": "all_fields_confirmed" if all_confirmed and validation["complete"]
                 else "partial_human_confirmations",
        "finalized": False,
        "fields": {key: value for key, value in fields.items()
                   if not key.startswith("row:")},
        "line_items": line_items,
        "incomplete_fields": validation["incomplete"],
        "validation_errors": validation["errors"],
        "accounting_export": False,
    }
