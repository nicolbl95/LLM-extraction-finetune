#!/usr/bin/env python3
"""Generate the requested assertion audit against the explicitly recorded Git base."""

import argparse
import ast
import difflib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "137b643"
RENAMES = {
    "test_cli_persists_safe_payload_and_raw_failures_without_inflating_metrics":
        "test_cli_persists_grounded_payload_and_raw_failures_without_inflating_metrics",
}
# "Reduced" refers to coverage of the old behavior, not an assertion that the
# replacement contract is safer. Removed exception/prompt guarantees are disclosed.
REDUCED = {
    "test_self_correction_preserves_decimal_quantity_and_injects_exact_pydantic_error",
    "test_arithmetic_retry_prioritizes_unit_price_for_a_french_decimal_quantity",
    "test_self_correction_handles_missing_fields_and_malformed_json",
    "test_no_retry_for_model_failures_or_truncated_responses",
    "test_second_arithmetic_inconsistency_is_displayed_without_a_third_attempt",
    "test_failed_correction_preserves_only_recoverable_fields_of_first_json",
    "test_model_error_is_not_hidden_without_a_valid_fallback",
    "test_rejects_invalid_json_constants_and_truncated_responses",
    "test_irrecoverable_fields_are_null_in_partial_json_display",
    "test_errors_clear_previous_json_and_success_is_validated",
    "test_document_and_adapter_preparation_precede_gpu_inference",
    "test_all_initial_line_and_global_anomalies_survive_successful_correction",
    "test_correction_uses_original_prompt_error_and_no_gold",
    "test_correction_arithmetic_warnings_remain_failures_for_scoring",
    "test_hallucinated_quantity_retries_with_operands_line_and_invoice_totals",
    "test_invoice_sum_mismatch_automatically_triggers_correction",
    "test_corrective_prompt_preserves_all_pydantic_line_errors",
    "test_two_invalid_responses_never_trigger_a_third_pass",
    "test_failed_correction_recovers_valid_fields_of_an_initial_partial_object",
    "test_cli_persists_safe_payload_and_raw_failures_without_inflating_metrics",
    "test_self_correction_recovers_exact_golden_outputs",
}
REASONS = {
    "test_ocr_failure_is_explicit": "Le faux OCR cible image_to_data/TSV plutôt que l'ancienne API texte ; les échecs explicites restent vérifiés.",
    "test_validated_result_uses_model_not_dataset": "Le résultat attendu doit être présent dans le document injecté, pas seulement dans la réponse du modèle.",
    "test_self_correction_preserves_decimal_quantity_and_injects_exact_pydantic_error": "L'OCR suffit maintenant sans seconde passe ; l'ancien contrat de transmission intégrale de Pydantic n'est plus exercé ici.",
    "test_arithmetic_retry_prioritizes_unit_price_for_a_french_decimal_quantity": "Le prix est relu dans les preuves sans appel correctif ; les assertions sur l'ancien prompt et son écart exact disparaissent.",
    "test_self_correction_handles_missing_fields_and_malformed_json": "Une sortie modèle malformée peut être remplacée par des preuves complètes ; l'ancienne seconde passe systématique n'est plus testée.",
    "test_no_retry_for_model_failures_or_truncated_responses": "L'exception propagée est remplacée par un repli OCR journalisé ; la garantie d'exception de l'ancien contrat est supprimée.",
    "test_second_arithmetic_inconsistency_is_displayed_without_a_third_attempt": "La valeur documentaire remplace le total erroné dès la première passe, sans tester la sélection d'un second JSON incohérent.",
    "test_failed_correction_preserves_only_recoverable_fields_of_first_json": "Les preuves récupèrent le résultat sans seconde passe ; cette boucle n'exerce donc plus réellement les erreurs de sa liste de secondes réponses.",
    "test_model_error_is_not_hidden_without_a_valid_fallback": "L'ancien échec propagé laisse place à un export sans valeurs non ancrées et à des logs ; le contrat d'exception n'est plus garanti.",
    "test_rejects_invalid_json_constants_and_truncated_responses": "Le rejet du modèle n'interdit plus un export partiel sécurisé ; l'exception n'est plus requise et le test de valeurs nulles peut être vide.",
    "test_arithmetic_inconsistencies_display_json_with_manual_review_warning": "Une incohérence imprimée reste fidèle dans le JSON et apparaît dans le rapport documentaire, au lieu d'annuler des lectures certaines.",
    "test_corrected_anomalies_remain_visible_above_json": "Les erreurs du modèle ne figurent plus comme anomalies de facture ; le test vérifie leur absence du rapport principal.",
    "test_supplier_and_decimal_normalizations_are_visible_without_retry": "Les valeurs normalisées sont désormais accompagnées d'un document source explicite.",
    "test_irrecoverable_fields_are_null_in_partial_json_display": "Le rapport visible résume la neutralisation ; l'ancien chemin de champ est déplacé aux détails et le nombre d'appels accepte maintenant <= 2.",
    "test_errors_clear_previous_json_and_success_is_validated": "Un OCR indisponible produit un résultat non vérifié plutôt qu'une exception brute ; les détails exacts de l'ancien message ne sont plus testés.",
    "test_document_and_adapter_preparation_precede_gpu_inference": "Le chargement des poids est différé pour autoriser le repli OCR ; l'appel préparatoire aux adaptateurs n'est plus exigé.",
    "test_zerogpu_registers_inference_at_import": "Le mock du décorateur vérifie l'argument duration dynamique au lieu de l'ancienne décoration sans argument.",
    "test_all_initial_line_and_global_anomalies_survive_successful_correction": "Le test couvre la récupération OCR et le diagnostic technique, plus l'historique de deux inférences sur un texte sans preuves.",
    "test_correction_uses_original_prompt_error_and_no_gold": "Un JSON malformé est recoupé sans seconde passe ; les assertions sur l'ancien échange et ses tokens cumulés sont remplacées.",
    "test_correction_arithmetic_warnings_remain_failures_for_scoring": "Un document incomplet conserve son total imprimé mais pas sa quantité absente ; la seconde passe est ciblée.",
    "test_hallucinated_quantity_retries_with_operands_line_and_invoice_totals": "La réponse brute reste fausse pour le scoring, tandis que l'export est récupéré sans second appel ; l'ancien prompt n'est plus testé.",
    "test_invoice_sum_mismatch_automatically_triggers_correction": "Le sous-total documentaire est conservé sans allouer de GPU supplémentaire ; l'ancien déclenchement obligatoire disparaît.",
    "test_corrective_prompt_preserves_all_pydantic_line_errors": "La fonction de prompt est testée directement, pas son branchement effectif dans une boucle d'inférence.",
    "test_failed_correction_keeps_structural_fallback_with_explicit_error": "Le second appel porte une ligne OCR incomplète ; les réponses brutes ne sont plus un export final de repli.",
    "test_two_invalid_responses_never_trigger_a_third_pass": "Sans preuves, aucune seconde passe inutile n'est lancée ; la borne reste testée mais le scénario de deux mauvaises réponses disparaît.",
    "test_failed_correction_recovers_valid_fields_of_an_initial_partial_object": "Seuls les champs documentés sont récupérés sans second appel ; l'ancien échec de seconde passe n'est plus exercé.",
    "test_cli_persists_safe_payload_and_raw_failures_without_inflating_metrics": "Renommage et export récupéré plutôt que null ; le test garde l'échec des métriques du modèle brut mais n'exerce plus deux passes.",
    "test_lora_generation_slices_prompt_and_checks_budget": "Une assertion supplémentaire exige temperature=0 ; aucune assertion retirée.",
    "test_self_correction_recovers_exact_golden_outputs": "Les sorties Golden exactes restent exigées, mais le texte suffit en un appel ; ce test ne couvre plus une vraie seconde passe.",
}


def functions(text):
    return {node.name: node for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def assertions(node):
    values = []
    for part in ast.walk(node):
        if isinstance(part, ast.Assert):
            values.append(ast.unparse(part))
        elif isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and part.func.attr.startswith("assert"):
            values.append(ast.unparse(part))
    return values


def generate():
    names = subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "tests"], cwd=ROOT, text=True).splitlines()
    records = []
    for name in names:
        if not name.endswith(".py"):
            continue
        old = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=ROOT, text=True)
        new = (ROOT / name).read_text()
        previous, current = functions(old), functions(new)
        for key, before in previous.items():
            after_name = RENAMES.get(key, key)
            after = current.get(after_name)
            if after is not None and ast.dump(before) == ast.dump(after):
                continue
            records.append((name, key, after_name if after else None, assertions(before), assertions(after) if after else []))
    lines = [
        "# Audit des tests existants modifiés",
        "",
        f"Comparaison reproductible : `{BASE}` → arbre de travail. "
        "Cet audit comprend les passes locales précédentes, pas seulement le dernier changement.",
        "",
        "Les anciennes et nouvelles assertions ci-dessous sont extraites de l'AST, sans paraphrase. "
        "Les assertions inchangées sont omises du diff ; les changements de fixtures sont explicités par la justification.",
        "",
        "**Affaiblissement** signifie une couverture historique retirée ou moins stricte, même si le nouveau contrat est intentionnel. "
        "Les nouveaux tests de quotas, ancrage et rapport ne restaurent pas automatiquement le contrat ancien.",
        "",
        "| Fichier | Test ancien → actuel | Couverture ancienne réduite ? | Justification |",
        "| --- | --- | --- | --- |",
    ]
    for path, old, new, _, _ in records:
        label = f"`{old}`" + (f" → `{new}`" if new and new != old else "")
        reduced = "OUI" if old in REDUCED else "Non (contrat modifié ou vérification renforcée)"
        lines.append(f"| [{path}](../{path}) | {label} | {reduced} | {REASONS.get(old, 'À examiner : changement non classé.')} |")
    lines.extend(["", f"Total : **{len(records)} tests existants modifiés/renommés**, "
                  f"**{sum(new is None for _, _, new, _, _ in records)} supprimé sans remplacement**.", ""])
    for path, old, new, before, after in records:
        lines.extend([f"## {path} — {old}", "", "**Anciennes assertions (-) / nouvelles assertions (+)** :", "", "```diff"])
        delta = list(difflib.unified_diff(before, after, fromfile=BASE, tofile="working-tree", lineterm=""))
        lines.extend(delta or ["Assertions identiques ; seuls le scénario, les données ou le mock changent."])
        lines.extend(["```", ""])
    lines.extend([
        "## Assertions affaiblies — regroupement par passe",
        "",
        "Les révisions intermédiaires locales n'ont pas été commitées : le détail AST ci-dessus "
        "compare le seul socle Git `137b643`. Les groupes ci-dessous reconstituent les passes "
        "fonctionnelles, pas des commits intermédiaires imaginaires.",
        "",
        "### Passe ancrage / suppression du repli historique",
        "",
    ])
    for path, old, new, before, after in records:
        if old in REDUCED:
            lines.append(f"- `{path}::{old}` : **couverture ancienne réduite** — {REASONS[old]}")
    lines.extend([
        "",
        "### Passe stress et qualification OCR",
        "",
        "- Aucun des 29 tests HEAD n'est supprimé sans remplacement. Les tests de stress "
        "sont nouveaux ; ils ne rétablissent pas l'ancien contrat de seconde passe des 21 tests "
        "signalés ci-dessus. La qualification globale pouvait rendre un document entièrement "
        "non vérifié : la couverture forte reste l'absence de chiffres faux, pas la disponibilité.",
        "",
        "### Passe récupération régionale / prétraitements",
        "",
        "- `test_image_ocr_is_x2_grayscale_tsv` : ancienne taille finale `(40,60)` remplacée "
        "par l'existence d'images ×2 et ×3 et la vérification des niveaux de gris pour tous les appels ; "
        "ce n'est pas un relâchement de sûreté mais le contrat d'un seul appel final disparaît.",
        "- `test_numbers_are_not_guessed` : le format groupé valide `1.234,56` est accepté, "
        "avec une assertion positive exacte ; `1.23,56` reste refusé. Domaine d'entrée élargi explicitement.",
        "- Le seuil 60→40 est un **assouplissement mesuré de la confiance**, pas une preuve "
        "de sûreté globale ; les tests de consensus indépendant et de veto textuel sont ajoutés.",
        "",
        "### Passe latence / données réelles / préparation du déploiement",
        "",
        "- Aucun résultat de scans réels n'est inventé. Les tests de budget utilisent "
        "une horloge contrôlée ; les mesures Tesseract sont séparées et réelles.",
        "- Le chemin rapide doit garder les mêmes assertions de sûreté et le budget doit "
        "rendre des données non vérifiées, jamais des valeurs par défaut ; toute adaptation "
        "de test nécessaire à ce chemin doit être décrite dans l'appendice ci-dessous.",
        "- Aucun des 29 tests historiques comparés à HEAD n'a été supprimé ou affaibli "
        "supplémentairement par cette passe. Les nouveaux tests de budget vérifient délai partagé, "
        "interdiction de lancer une étape après épuisement, borne transmise à Tesseract et absence "
        "de reparsing des quantités OCR rejetées dans un PDF.",
        "- `test_clear_first_pair_never_runs_extra_families`, créé dans cette passe : "
        "une première assertion attendait seulement `deskew` ; elle est remplacée par "
        "`['deskew', 'clahe']` avec deux appels [4,6], pour **renforcer** le recoupement indépendant. "
        "Le premier candidat à deux lectures du même prétraitement n'est pas la version retenue.",
        "- Les huit tests du script de scans réels n'utilisent que des PDF de test générés "
        "et n'affirment aucune validation sur un document réel indépendant.",
        "",
        "## Tests nouveaux au regard de HEAD mais déjà présents dans l'arbre local",
        "",
        "- `tests/test_grounding.py::test_numbers_are_not_guessed` : `1.234,56` n'est plus rejeté "
        "comme ambigu ; `1.23,56` reste refusé et un test positif exige exactement `Decimal('1234.56')`. "
        "C'est un élargissement volontaire et vérifié, pas une suppression de contrôle des chiffres.",
        "- `tests/test_grounding.py::test_real_tesseract_evidence_with_simulated_wrong_model_outputs_on_all_three_images` : "
        "l'exclusion des pourcentages vise les trois champs quantité/prix/montant, pas les colonnes explicites de TVA/remise.",
        "- Passe récupération des null : `tests/test_grounding.py::test_image_ocr_is_x2_grayscale_tsv` "
        "ne suppose plus un seul appel OCR ×2 ; il exige des appels ×2 **et ×3**, tous en niveaux de gris, "
        "avec la même langue et sortie TSV. L'ancienne assertion sur la taille d'un seul dernier appel "
        "est remplacée par une vérification des tailles effectivement utilisées.",
        "- Les anciens tests qui exigeaient le rejet global des images insuffisantes restent en place ; "
        "ils portent sur les preuves insuffisantes synthétiques, pas sur une désactivation des garde-fous. "
        "Aucun test de sûreté n'est retiré pour améliorer le nombre de null.",
        "- Historique des suites : rapport documentaire 177 tests, stress 237, récupération "
        "249, latence/scans réels 271 (plus 101 sous-tests et 3 slow exclus). "
        "Les scénarios historiques adaptés ne gardent pas forcément leurs anciennes assertions : "
        "les 21 réductions de couverture ci-dessus restent explicitement signalées.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(generate(), encoding="utf-8")


if __name__ == "__main__":
    main()
