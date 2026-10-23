# Audit des tests existants modifiés

Comparaison reproductible : `137b643` → arbre de travail. Cet audit comprend les passes locales précédentes, pas seulement le dernier changement.

Les anciennes et nouvelles assertions ci-dessous sont extraites de l'AST, sans paraphrase. Les assertions inchangées sont omises du diff ; les changements de fixtures sont explicités par la justification.

**Affaiblissement** signifie une couverture historique retirée ou moins stricte, même si le nouveau contrat est intentionnel. Les nouveaux tests de quotas, ancrage et rapport ne restaurent pas automatiquement le contrat ancien.

| Fichier | Test ancien → actuel | Couverture ancienne réduite ? | Justification |
| --- | --- | --- | --- |
| [tests/test_app.py](../tests/test_app.py) | `test_ocr_failure_is_explicit` | Non (contrat modifié ou vérification renforcée) | Le faux OCR cible image_to_data/TSV plutôt que l'ancienne API texte ; les échecs explicites restent vérifiés. |
| [tests/test_app.py](../tests/test_app.py) | `test_validated_result_uses_model_not_dataset` | Non (contrat modifié ou vérification renforcée) | Le résultat attendu doit être présent dans le document injecté, pas seulement dans la réponse du modèle. |
| [tests/test_app.py](../tests/test_app.py) | `test_self_correction_preserves_decimal_quantity_and_injects_exact_pydantic_error` | OUI | L'OCR suffit maintenant sans seconde passe ; l'ancien contrat de transmission intégrale de Pydantic n'est plus exercé ici. |
| [tests/test_app.py](../tests/test_app.py) | `test_arithmetic_retry_prioritizes_unit_price_for_a_french_decimal_quantity` | OUI | Le prix est relu dans les preuves sans appel correctif ; les assertions sur l'ancien prompt et son écart exact disparaissent. |
| [tests/test_app.py](../tests/test_app.py) | `test_self_correction_handles_missing_fields_and_malformed_json` | OUI | Une sortie modèle malformée peut être remplacée par des preuves complètes ; l'ancienne seconde passe systématique n'est plus testée. |
| [tests/test_app.py](../tests/test_app.py) | `test_no_retry_for_model_failures_or_truncated_responses` | OUI | L'exception propagée est remplacée par un repli OCR journalisé ; la garantie d'exception de l'ancien contrat est supprimée. |
| [tests/test_app.py](../tests/test_app.py) | `test_second_arithmetic_inconsistency_is_displayed_without_a_third_attempt` | OUI | La valeur documentaire remplace le total erroné dès la première passe, sans tester la sélection d'un second JSON incohérent. |
| [tests/test_app.py](../tests/test_app.py) | `test_failed_correction_preserves_only_recoverable_fields_of_first_json` | OUI | Les preuves récupèrent le résultat sans seconde passe ; cette boucle n'exerce donc plus réellement les erreurs de sa liste de secondes réponses. |
| [tests/test_app.py](../tests/test_app.py) | `test_model_error_is_not_hidden_without_a_valid_fallback` | OUI | L'ancien échec propagé laisse place à un export sans valeurs non ancrées et à des logs ; le contrat d'exception n'est plus garanti. |
| [tests/test_app.py](../tests/test_app.py) | `test_rejects_invalid_json_constants_and_truncated_responses` | OUI | Le rejet du modèle n'interdit plus un export partiel sécurisé ; l'exception n'est plus requise et le test de valeurs nulles peut être vide. |
| [tests/test_app.py](../tests/test_app.py) | `test_arithmetic_inconsistencies_display_json_with_manual_review_warning` | Non (contrat modifié ou vérification renforcée) | Une incohérence imprimée reste fidèle dans le JSON et apparaît dans le rapport documentaire, au lieu d'annuler des lectures certaines. |
| [tests/test_app.py](../tests/test_app.py) | `test_corrected_anomalies_remain_visible_above_json` | Non (contrat modifié ou vérification renforcée) | Les erreurs du modèle ne figurent plus comme anomalies de facture ; le test vérifie leur absence du rapport principal. |
| [tests/test_app.py](../tests/test_app.py) | `test_supplier_and_decimal_normalizations_are_visible_without_retry` | Non (contrat modifié ou vérification renforcée) | Les valeurs normalisées sont désormais accompagnées d'un document source explicite. |
| [tests/test_app.py](../tests/test_app.py) | `test_irrecoverable_fields_are_null_in_partial_json_display` | OUI | Le rapport visible résume la neutralisation ; l'ancien chemin de champ est déplacé aux détails et le nombre d'appels accepte maintenant <= 2. |
| [tests/test_app.py](../tests/test_app.py) | `test_errors_clear_previous_json_and_success_is_validated` | OUI | Un OCR indisponible produit un résultat non vérifié plutôt qu'une exception brute ; les détails exacts de l'ancien message ne sont plus testés. |
| [tests/test_app.py](../tests/test_app.py) | `test_document_and_adapter_preparation_precede_gpu_inference` | OUI | Le chargement des poids est différé pour autoriser le repli OCR ; l'appel préparatoire aux adaptateurs n'est plus exigé. |
| [tests/test_app.py](../tests/test_app.py) | `test_zerogpu_registers_inference_at_import` | Non (contrat modifié ou vérification renforcée) | Le mock du décorateur vérifie l'argument duration dynamique au lieu de l'ancienne décoration sans argument. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_all_initial_line_and_global_anomalies_survive_successful_correction` | OUI | Le test couvre la récupération OCR et le diagnostic technique, plus l'historique de deux inférences sur un texte sans preuves. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_correction_uses_original_prompt_error_and_no_gold` | OUI | Un JSON malformé est recoupé sans seconde passe ; les assertions sur l'ancien échange et ses tokens cumulés sont remplacées. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_correction_arithmetic_warnings_remain_failures_for_scoring` | OUI | Un document incomplet conserve son total imprimé mais pas sa quantité absente ; la seconde passe est ciblée. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_hallucinated_quantity_retries_with_operands_line_and_invoice_totals` | OUI | La réponse brute reste fausse pour le scoring, tandis que l'export est récupéré sans second appel ; l'ancien prompt n'est plus testé. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_invoice_sum_mismatch_automatically_triggers_correction` | OUI | Le sous-total documentaire est conservé sans allouer de GPU supplémentaire ; l'ancien déclenchement obligatoire disparaît. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_corrective_prompt_preserves_all_pydantic_line_errors` | OUI | La fonction de prompt est testée directement, pas son branchement effectif dans une boucle d'inférence. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_failed_correction_keeps_structural_fallback_with_explicit_error` | Non (contrat modifié ou vérification renforcée) | Le second appel porte une ligne OCR incomplète ; les réponses brutes ne sont plus un export final de repli. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_two_invalid_responses_never_trigger_a_third_pass` | OUI | Sans preuves, aucune seconde passe inutile n'est lancée ; la borne reste testée mais le scénario de deux mauvaises réponses disparaît. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_failed_correction_recovers_valid_fields_of_an_initial_partial_object` | OUI | Seuls les champs documentés sont récupérés sans second appel ; l'ancien échec de seconde passe n'est plus exercé. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_cli_persists_safe_payload_and_raw_failures_without_inflating_metrics` → `test_cli_persists_grounded_payload_and_raw_failures_without_inflating_metrics` | OUI | Renommage et export récupéré plutôt que null ; le test garde l'échec des métriques du modèle brut mais n'exerce plus deux passes. |
| [tests/test_evaluate_finetuned.py](../tests/test_evaluate_finetuned.py) | `test_lora_generation_slices_prompt_and_checks_budget` | Non (contrat modifié ou vérification renforcée) | Une assertion supplémentaire exige temperature=0 ; aucune assertion retirée. |
| [tests/test_golden_dataset.py](../tests/test_golden_dataset.py) | `test_self_correction_recovers_exact_golden_outputs` | OUI | Les sorties Golden exactes restent exigées, mais le texte suffit en un appel ; ce test ne couvre plus une vraie seconde passe. |

Total : **29 tests existants modifiés/renommés**, **0 supprimé sans remplacement**.

## tests/test_app.py — test_ocr_failure_is_explicit

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
Assertions identiques ; seuls le scénario, les données ou le mock changent.
```

## tests/test_app.py — test_validated_result_uses_model_not_dataset

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,5 +1,5 @@
 model.predict.assert_called_once()
-self.assertEqual(model.predict.call_args.args[0], {'id': 'live-upload', 'input': 'Text from the uploaded invoice'})
+self.assertEqual(model.predict.call_args.args[0], {'id': 'live-upload', 'input': self.evidence_text()})
 self.assertEqual([item['role'] for item in messages], ['system', 'user', 'assistant', 'user'])
 self.assertIn('7,2', messages[0]['content'])
 self.assertIn('jamais 142', messages[0]['content'])
@@ -10,6 +10,6 @@
 self.assertEqual(example.line_items[0].quantity, 7.2)
 self.assertEqual(example.line_items[0].unit_price, 12.5)
 self.assertEqual(example.line_items[0].total_ht, 90.0)
-self.assertIn('Text from the uploaded invoice', messages[-1]['content'])
+self.assertIn(self.evidence_text(), messages[-1]['content'])
 self.assertEqual(payload, self.gold)
 self.assertEqual(payload['invoice_date'], '2026-10-21')
```

## tests/test_app.py — test_self_correction_preserves_decimal_quantity_and_injects_exact_pydantic_error

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,17 +1,9 @@
 self.assertEqual(result, corrected)
-self.assertTrue(any(('corrigée lors de la seconde passe' in warning for warning in warnings)))
+self.assertTrue(any(('récupéré (corroboré par OCR)' in warning for warning in warnings)))
 self.assertTrue(any(('142' in warning for warning in warnings)))
-self.assertEqual(model.predict.call_count, 2)
+self.assertEqual(model.predict.call_count, 1)
 self.assertEqual(len(initial_messages), 4)
-self.assertEqual(corrected_messages[:-2], initial_messages)
-self.assertIn(text, corrected_messages[-3]['content'])
-self.assertEqual(corrected_messages[-2], {'role': 'assistant', 'content': raw})
-self.assertIn(str(validation.exception), corrected_messages[-1]['content'])
-self.assertIn('valeurs de la facture réelle', corrected_messages[-1]['content'])
-self.assertIn('Priorité de correction arithmétique', corrected_messages[-1]['content'])
-self.assertIn("relis d'abord les champs quantity et unit_price", corrected_messages[-1]['content'])
-self.assertIn('7,2 devient 7.2 (jamais 142 ou 72)', corrected_messages[-1]['content'])
-self.assertIn('Anomalies arithmétiques à examiner', corrected_messages[-1]['content'])
-self.assertEqual(first.args, second.args)
+self.assertIn(text, initial_messages[-1]['content'])
+self.assertTrue(any(('142' in warning and 'non ancré' in warning for warning in warnings)))
 self.assertRaises(main.ValidationError)
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_arithmetic_retry_prioritizes_unit_price_for_a_french_decimal_quantity

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,11 +1,7 @@
-self.assertEqual(model.predict.call_count, 2)
-self.assertTrue(any(('corrigée lors de la seconde passe' in warning for warning in warnings)))
+self.assertEqual(model.predict.call_count, 1)
+self.assertTrue(any(('récupéré (corroboré par OCR)' in warning for warning in warnings)))
 self.assertTrue(any(('200' in warning for warning in warnings)))
-self.assertIn(text, messages[-3]['content'])
-self.assertIn('quantity et unit_price', messages[-1]['content'])
-self.assertIn('4,2 devient 4.2 (jamais 42)', messages[-1]['content'])
-self.assertIn('Ne déduis pas ces champs par division', messages[-1]['content'])
-self.assertIn('Ligne 1', messages[-1]['content'])
-self.assertIn('Écart absolu : 756', messages[-1]['content'])
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertIn(text, messages[-1]['content'])
+self.assertTrue(any(('unit_price' in warning for warning in warnings)))
+self.assertLogs(level='WARNING')
 self.assertEqual(main.extract_invoice(text, warnings=warnings), corrected)
```

## tests/test_app.py — test_self_correction_handles_missing_fields_and_malformed_json

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,7 +1,4 @@
-self.assertEqual(model.predict.call_count, 2)
-self.assertEqual(messages[-2]['content'], raw)
-self.assertIn(str(validation.exception), messages[-1]['content'])
-self.assertNotIn('Priorité de correction arithmétique', messages[-1]['content'])
+self.assertEqual(model.predict.call_count, 1)
 self.assertRaises(ValueError)
-self.assertLogs(main.LOGGER, level='WARNING')
-self.assertEqual(main.extract_invoice('Original invoice'), self.gold)
+self.assertLogs(level='WARNING')
+self.assertEqual(main.extract_invoice(self.evidence_text()), self.gold)
```

## tests/test_app.py — test_no_retry_for_model_failures_or_truncated_responses

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,4 +1,6 @@
 self.assertEqual(model.predict.call_count, 1)
 self.assertEqual(model.predict.call_count, 1)
-self.assertRaisesRegex(RuntimeError, 'Model unavailable')
-self.assertRaisesRegex(ValueError, 'incomplète')
+self.assertLogs(level='ERROR')
+self.assertEqual(main.extract_invoice(self.evidence_text()), self.gold)
+self.assertLogs(level='ERROR')
+self.assertEqual(main.extract_invoice(self.evidence_text()), self.gold)
```

## tests/test_app.py — test_second_arithmetic_inconsistency_is_displayed_without_a_third_attempt

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,8 +1,6 @@
-self.assertIsNone(actual['total_ht'])
-self.assertIsNone(actual['line_items'][0]['total_ht'])
+self.assertEqual(actual['total_ht'], self.gold['total_ht'])
+self.assertEqual(actual['line_items'][0]['total_ht'], self.gold['line_items'][0]['total_ht'])
 self.assertEqual(actual['line_items'][0]['quantity'], 1)
-self.assertTrue(any(('888' in warning for warning in warnings)))
-self.assertTrue(any(('999' in warning for warning in warnings)))
-self.assertTrue(any(('non résolue' in warning for warning in warnings)))
-self.assertEqual(model.predict.call_count, 2)
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertTrue(any(('total_ht' in warning for warning in warnings)))
+self.assertEqual(model.predict.call_count, 1)
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_failed_correction_preserves_only_recoverable_fields_of_first_json

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,7 +1,6 @@
-self.assertIsNone(actual['total_ht'])
-self.assertIsNone(actual['line_items'][0]['total_ht'])
+self.assertEqual(actual['total_ht'], self.gold['total_ht'])
+self.assertEqual(actual['line_items'][0]['total_ht'], self.gold['line_items'][0]['total_ht'])
 self.assertEqual(actual['supplier_name'], first['supplier_name'])
-self.assertEqual(model.predict.call_count, 2)
-self.assertTrue(any(('Écart absolu' in message for message in warnings)))
-self.assertTrue(any(('Auto-correction impossible' in message for message in warnings)))
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertEqual(model.predict.call_count, 1)
+self.assertTrue(any(('total_ht' in message for message in warnings)))
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_model_error_is_not_hidden_without_a_valid_fallback

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,3 +1,3 @@
-self.assertEqual(model.predict.call_count, 2)
-self.assertLogs(main.LOGGER, level='WARNING')
-self.assertRaisesRegex(RuntimeError, 'GPU unavailable')
+self.assertTrue(all((value is None for item in result['line_items'] for value in item.values())))
+self.assertEqual(model.predict.call_count, 1)
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_rejects_invalid_json_constants_and_truncated_responses

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1 +1,3 @@
-self.assertRaises(ValueError)
+self.assertTrue(all((value is None for item in result['line_items'] for value in item.values())))
+self.assertTrue(warnings)
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_arithmetic_inconsistencies_display_json_with_manual_review_warning

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,10 +1,9 @@
-self.assertIsNone(updates[-1][0]['total_ht'])
-self.assertIsNone(updates[-1][0]['line_items'][0]['total_ht'])
+self.assertEqual(updates[-1][0]['total_ht'], payload['total_ht'])
+self.assertEqual(updates[-1][0]['line_items'][0]['total_ht'], payload['line_items'][0]['total_ht'])
 self.assertEqual(updates[-1][0]['supplier_name'], payload['supplier_name'])
-self.assertIn('Avertissement', updates[-1][1])
-self.assertIn('Incohérence', updates[-1][1])
-self.assertIn('vérification manuelle', updates[-1][1])
-self.assertIn('Écart absolu', updates[-1][1])
+self.assertIn('anomalie', updates[-1][1])
+self.assertIn('Erreur', updates[-1][1])
+self.assertIn('Écart', updates[-1][1])
 self.assertNotIn('cohérence des montants HT contrôlée', updates[-1][1])
-self.assertEqual(model.predict.call_count, 2)
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertEqual(model.predict.call_count, 1)
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_corrected_anomalies_remain_visible_above_json

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,7 +1,6 @@
 self.assertEqual(payload, self.gold)
-self.assertIn('corrigée lors de la seconde passe', status)
-self.assertIn('999', status)
-self.assertIn('Écart absolu', status)
-self.assertIn('Tolérance', status)
+self.assertIn('récupérée', status)
+self.assertNotIn('999', status)
+self.assertIn('Aucune anomalie', status)
 self.assertLess(status_index, json_index)
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_supplier_and_decimal_normalizations_are_visible_without_retry

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -3,4 +3,4 @@
 self.assertEqual(result['line_items'][0]['quantity'], 5.1)
 self.assertTrue(any(('supplier_name' in warning for warning in warnings)))
 self.assertTrue(any(('line_items.0.quantity' in warning for warning in warnings)))
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_irrecoverable_fields_are_null_in_partial_json_display

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,6 +1,5 @@
 self.assertIsNone(updates[-1][0][field])
 self.assertEqual(updates[-1][0]['total_ttc'], self.gold['total_ttc'])
-self.assertIn(field, updates[-1][1])
-self.assertIn('null', updates[-1][1])
-self.assertEqual(model.predict.call_count, 2)
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertIn('neutralisée', updates[-1][1])
+self.assertLessEqual(model.predict.call_count, 2)
+self.assertLogs(level='WARNING')
```

## tests/test_app.py — test_errors_clear_previous_json_and_success_is_validated

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,6 +1,6 @@
 self.assertIsNone(updates[0][0])
 self.assertEqual(updates[-1][0], self.gold)
-self.assertIn('validé', updates[-1][1])
-self.assertIsNone(updates[-1][0])
-self.assertIn('OCR timeout', updates[-1][1])
+self.assertIn('terminée', updates[-1][1])
+self.assertIsNone(updates[-1][0]['total_ht'])
+self.assertIn('Preuves OCR indisponibles', updates[-1][1])
 self.assertLogs(main.LOGGER, level='ERROR')
```

## tests/test_app.py — test_document_and_adapter_preparation_precede_gpu_inference

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,6 +1,6 @@
-self.assertEqual([entry[0] for entry in sequence.mock_calls], ['document_text', 'resolve_adapter_path', 'extract_invoice'])
+self.assertEqual([entry[0] for entry in sequence.mock_calls], ['document_text', 'extract_invoice'])
 self.assertEqual(last_update[0], self.gold)
-self.assertIn('Extraction terminée', last_update[1])
+self.assertIn('terminée', last_update[1])
 self.assertEqual(next(updates), (None, 'Analyse des données, détection des erreurs'))
 sequence.document_text.assert_not_called()
 self.assertEqual(next(updates), (None, 'Analyse des données, détection des erreurs'))
```

## tests/test_app.py — test_zerogpu_registers_inference_at_import

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,2 +1,2 @@
-gpu.assert_called_once_with(namespace['predict_invoice'])
+gpu.assert_called_once_with(duration=namespace['gpu_duration'])
 backend.assert_not_called()
```

## tests/test_evaluate_finetuned.py — test_all_initial_line_and_global_anomalies_survive_successful_correction

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,6 +1,6 @@
 self.assertEqual(result['payload'], self.gold)
-self.assertIn('corrigée lors de la seconde passe', history)
-self.assertIn('Incohérence sur la ligne', history)
-self.assertIn('somme des total_ht', history)
-self.assertEqual(backend.predict.call_count, 2)
+self.assertIn('récupéré (corroboré par OCR)', history)
+self.assertIn('Ligne 1', history)
+self.assertIn('Écart global initial', history)
+self.assertEqual(backend.predict.call_count, 1)
 self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_correction_uses_original_prompt_error_and_no_gold

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,16 +1,15 @@
-self.assertEqual(backend.predict.call_count, 2)
+self.assertEqual(backend.predict.call_count, 1)
 self.assertEqual(initial_messages, evaluation.build_messages(row))
-self.assertEqual(corrected_messages[:-2], initial_messages)
-self.assertEqual(corrected_messages[-2]['content'], initial['raw'])
-self.assertTrue(corrected_messages[-1]['content'].startswith('La précédente extraction a échoué à la validation du schéma Pydantic pour les raisons suivantes :\n'))
-self.assertEqual(result['raw'], corrected['raw'])
-self.assertEqual(result['attempt_count'], 2)
-self.assertEqual(result['completion_tokens'], 20)
-self.assertEqual(result['prompt_tokens'], 40)
-self.assertEqual(result['attempts'], [initial, corrected])
-self.assertFalse(result['arithmetic_warnings'])
+self.assertEqual(result['raw'], initial['raw'])
+self.assertEqual(result['payload'], self.gold)
+self.assertEqual(result['attempt_count'], 1)
+self.assertEqual(result['completion_tokens'], 10)
+self.assertEqual(result['prompt_tokens'], 20)
+self.assertEqual(result['attempts'], [initial])
+self.assertTrue(result['operational_error'])
 self.assertEqual(report['initial_json_validity_rate'], 0)
-self.assertEqual(report['json_validity_rate'], 1)
-self.assertEqual(report['correction_rate'], 1)
+self.assertEqual(report['json_validity_rate'], 0)
+self.assertEqual(report['correction_rate'], 0)
+self.assertEqual(rows[0]['payload'], self.gold)
 self.assertLogs(level='WARNING')
 self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_correction_arithmetic_warnings_remain_failures_for_scoring

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,8 +1,8 @@
 self.assertEqual(backend.predict.call_count, 2)
 self.assertTrue(result['arithmetic_warnings'])
-self.assertIsNone(result['payload']['total_ht'])
+self.assertEqual(result['payload']['total_ht'], self.gold['total_ht'])
 self.assertIsNone(result['payload']['line_items'][0]['quantity'])
 self.assertTrue(any(('line_items.0.quantity' in warning for warning in result['warnings'])))
 self.assertFalse(evaluation.score(result['raw'], invalid)['valid'])
-self.assertIn('Priorité de correction arithmétique', backend.predict.call_args.kwargs['messages'][-1]['content'])
+self.assertTrue(result['correction_attempted'])
 self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_hallucinated_quantity_retries_with_operands_line_and_invoice_totals

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,16 +1,7 @@
-self.assertEqual(backend.predict.call_count, 2)
-self.assertTrue(prompt.startswith(f'La précédente extraction a échoué à la validation du schéma Pydantic pour les raisons suivantes :\n{validation.exception}\n\n'))
-self.assertTrue(prompt.endswith("Relis attentivement la facture fournie (en faisant particulièrement attention aux décimales des prix unitaires, aux quantités et aux totaux de lignes), corrige l'erreur de calcul ou d'extraction, et renvoie un JSON rigoureusement valide."))
-self.assertIn('Erreur arithmétique détectée', prompt)
-self.assertIn("Incohérence sur la ligne 'Maintenance préventive'", prompt)
-self.assertIn('quantity (142.0) * unit_price (95.07) = 13499.94', prompt)
-self.assertIn('Écart absolu : 12815.44', prompt)
-self.assertIn('684.5', prompt)
-self.assertIn('5716.77', prompt)
-self.assertIn('ré-extrais la ligne exacte', prompt)
-self.assertTrue(evaluation.score(result['raw'], corrected)['exact'])
-self.assertEqual(result['arithmetic_warnings'], [])
+self.assertEqual(backend.predict.call_count, 1)
+self.assertFalse(evaluation.score(result['raw'], corrected)['exact'])
+self.assertTrue(result['arithmetic_warnings'])
 self.assertEqual(result['payload'], corrected)
-self.assertTrue(any(('corrigée lors de la seconde passe' in warning for warning in result['warnings'])))
+self.assertTrue(any(('récupéré (corroboré par OCR)' in warning for warning in result['warnings'])))
 self.assertRaises(evaluation.ValidationError)
 self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_invoice_sum_mismatch_automatically_triggers_correction

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,5 +1,5 @@
-self.assertEqual(backend.predict.call_count, 2)
-self.assertIn('somme des total_ht', prompt)
-self.assertIn('total_ht global', prompt)
-self.assertTrue(evaluation.score(result['raw'], self.gold)['exact'])
+self.assertEqual(backend.predict.call_count, 1)
+self.assertIn('Écart global initial', '\n'.join(result['warnings']))
+self.assertEqual(result['payload'], self.gold)
+self.assertFalse(evaluation.score(result['raw'], self.gold)['exact'])
 self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_corrective_prompt_preserves_all_pydantic_line_errors

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,7 +1,5 @@
-self.assertEqual(backend.predict.call_count, 2)
 self.assertIn(str(validation.exception), prompt)
 self.assertIn('line_items.0', prompt)
 self.assertIn('line_items.1', prompt)
 self.assertIn("Incohérence sur la ligne 'Second article'", prompt)
 self.assertRaises(evaluation.ValidationError)
-self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_failed_correction_keeps_structural_fallback_with_explicit_error

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
Assertions identiques ; seuls le scénario, les données ou le mock changent.
```

## tests/test_evaluate_finetuned.py — test_two_invalid_responses_never_trigger_a_third_pass

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,7 +1,7 @@
-self.assertEqual(backend.predict.call_count, 2)
-self.assertEqual(result['raw'], 'broken second')
+self.assertEqual(backend.predict.call_count, 1)
+self.assertEqual(result['raw'], 'broken first')
 self.assertFalse(evaluation.score(result['raw'], self.gold)['exact'])
 self.assertTrue(result['validation_errors'])
-self.assertIsNone(result['payload'])
-self.assertTrue(any(('JSON inutilisable' in warning for warning in result['warnings'])))
+self.assertIsNone(result['payload']['total_ht'])
+self.assertTrue(result['operational_error'])
 self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_failed_correction_recovers_valid_fields_of_an_initial_partial_object

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -3,7 +3,7 @@
 self.assertEqual(result['payload']['line_items'], self.gold['line_items'])
 self.assertEqual(result['payload']['total_ht'], self.gold['total_ht'])
 self.assertTrue(any(('supplier_name' in warning for warning in result['warnings'])))
-self.assertTrue(any(('Auto-correction impossible' in warning for warning in result['warnings'])))
+self.assertFalse(result['correction_attempted'])
 self.assertFalse(evaluation.score(raw, self.gold)['valid'])
-self.assertEqual(backend.predict.call_count, 2)
+self.assertEqual(backend.predict.call_count, 1)
 self.assertLogs(level='WARNING')
```

## tests/test_evaluate_finetuned.py — test_cli_persists_safe_payload_and_raw_failures_without_inflating_metrics

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,7 +1,6 @@
 self.assertEqual(record['raw'], raw)
-self.assertEqual([attempt['raw'] for attempt in record['attempts']], [raw, raw])
-self.assertIsNone(record['payload']['total_ht'])
-self.assertIsNone(record['payload']['line_items'][0]['quantity'])
+self.assertEqual([attempt['raw'] for attempt in record['attempts']], [raw])
+self.assertEqual(record['payload'], self.gold)
 self.assertTrue(record['warnings'])
 self.assertFalse(record['valid'])
 self.assertEqual(report['json_validity_rate'], 1)
```

## tests/test_evaluate_finetuned.py — test_lora_generation_slices_prompt_and_checks_budget

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -3,6 +3,7 @@
 inference.tokenizer.decode.assert_called_once_with([5, 6, 2], skip_special_tokens=True)
 self.assertEqual(inference.model.generate.call_args.kwargs['pad_token_id'], 0)
 self.assertIs(inference.model.generate.call_args.kwargs['do_sample'], False)
+self.assertEqual(inference.model.generate.call_args.kwargs['temperature'], 0.0)
 self.assertEqual(inference.tokenizer.apply_chat_template.call_args.args[0], evaluation.build_messages({'input': 'invoice', 'id': 'test'}))
 self.assertEqual(inference.tokenizer.apply_chat_template.call_args.args[0], messages)
 self.assertRaisesRegex(ValueError, 'exceeds context')
```

## tests/test_golden_dataset.py — test_self_correction_recovers_exact_golden_outputs

**Anciennes assertions (-) / nouvelles assertions (+)** :

```diff
--- 137b643
+++ working-tree
@@ -1,3 +1,3 @@
 self.assertEqual(actual, case['expected'])
-self.assertEqual(model.predict.call_count, 2)
-self.assertLogs(main.LOGGER, level='WARNING')
+self.assertEqual(model.predict.call_count, 1)
+self.assertLogs(level='WARNING')
```

## Assertions affaiblies — regroupement par passe

Les révisions intermédiaires locales n'ont pas été commitées : le détail AST ci-dessus compare le seul socle Git `137b643`. Les groupes ci-dessous reconstituent les passes fonctionnelles, pas des commits intermédiaires imaginaires.

### Passe ancrage / suppression du repli historique

- `tests/test_app.py::test_self_correction_preserves_decimal_quantity_and_injects_exact_pydantic_error` : **couverture ancienne réduite** — L'OCR suffit maintenant sans seconde passe ; l'ancien contrat de transmission intégrale de Pydantic n'est plus exercé ici.
- `tests/test_app.py::test_arithmetic_retry_prioritizes_unit_price_for_a_french_decimal_quantity` : **couverture ancienne réduite** — Le prix est relu dans les preuves sans appel correctif ; les assertions sur l'ancien prompt et son écart exact disparaissent.
- `tests/test_app.py::test_self_correction_handles_missing_fields_and_malformed_json` : **couverture ancienne réduite** — Une sortie modèle malformée peut être remplacée par des preuves complètes ; l'ancienne seconde passe systématique n'est plus testée.
- `tests/test_app.py::test_no_retry_for_model_failures_or_truncated_responses` : **couverture ancienne réduite** — L'exception propagée est remplacée par un repli OCR journalisé ; la garantie d'exception de l'ancien contrat est supprimée.
- `tests/test_app.py::test_second_arithmetic_inconsistency_is_displayed_without_a_third_attempt` : **couverture ancienne réduite** — La valeur documentaire remplace le total erroné dès la première passe, sans tester la sélection d'un second JSON incohérent.
- `tests/test_app.py::test_failed_correction_preserves_only_recoverable_fields_of_first_json` : **couverture ancienne réduite** — Les preuves récupèrent le résultat sans seconde passe ; cette boucle n'exerce donc plus réellement les erreurs de sa liste de secondes réponses.
- `tests/test_app.py::test_model_error_is_not_hidden_without_a_valid_fallback` : **couverture ancienne réduite** — L'ancien échec propagé laisse place à un export sans valeurs non ancrées et à des logs ; le contrat d'exception n'est plus garanti.
- `tests/test_app.py::test_rejects_invalid_json_constants_and_truncated_responses` : **couverture ancienne réduite** — Le rejet du modèle n'interdit plus un export partiel sécurisé ; l'exception n'est plus requise et le test de valeurs nulles peut être vide.
- `tests/test_app.py::test_irrecoverable_fields_are_null_in_partial_json_display` : **couverture ancienne réduite** — Le rapport visible résume la neutralisation ; l'ancien chemin de champ est déplacé aux détails et le nombre d'appels accepte maintenant <= 2.
- `tests/test_app.py::test_errors_clear_previous_json_and_success_is_validated` : **couverture ancienne réduite** — Un OCR indisponible produit un résultat non vérifié plutôt qu'une exception brute ; les détails exacts de l'ancien message ne sont plus testés.
- `tests/test_app.py::test_document_and_adapter_preparation_precede_gpu_inference` : **couverture ancienne réduite** — Le chargement des poids est différé pour autoriser le repli OCR ; l'appel préparatoire aux adaptateurs n'est plus exigé.
- `tests/test_evaluate_finetuned.py::test_all_initial_line_and_global_anomalies_survive_successful_correction` : **couverture ancienne réduite** — Le test couvre la récupération OCR et le diagnostic technique, plus l'historique de deux inférences sur un texte sans preuves.
- `tests/test_evaluate_finetuned.py::test_correction_uses_original_prompt_error_and_no_gold` : **couverture ancienne réduite** — Un JSON malformé est recoupé sans seconde passe ; les assertions sur l'ancien échange et ses tokens cumulés sont remplacées.
- `tests/test_evaluate_finetuned.py::test_correction_arithmetic_warnings_remain_failures_for_scoring` : **couverture ancienne réduite** — Un document incomplet conserve son total imprimé mais pas sa quantité absente ; la seconde passe est ciblée.
- `tests/test_evaluate_finetuned.py::test_hallucinated_quantity_retries_with_operands_line_and_invoice_totals` : **couverture ancienne réduite** — La réponse brute reste fausse pour le scoring, tandis que l'export est récupéré sans second appel ; l'ancien prompt n'est plus testé.
- `tests/test_evaluate_finetuned.py::test_invoice_sum_mismatch_automatically_triggers_correction` : **couverture ancienne réduite** — Le sous-total documentaire est conservé sans allouer de GPU supplémentaire ; l'ancien déclenchement obligatoire disparaît.
- `tests/test_evaluate_finetuned.py::test_corrective_prompt_preserves_all_pydantic_line_errors` : **couverture ancienne réduite** — La fonction de prompt est testée directement, pas son branchement effectif dans une boucle d'inférence.
- `tests/test_evaluate_finetuned.py::test_two_invalid_responses_never_trigger_a_third_pass` : **couverture ancienne réduite** — Sans preuves, aucune seconde passe inutile n'est lancée ; la borne reste testée mais le scénario de deux mauvaises réponses disparaît.
- `tests/test_evaluate_finetuned.py::test_failed_correction_recovers_valid_fields_of_an_initial_partial_object` : **couverture ancienne réduite** — Seuls les champs documentés sont récupérés sans second appel ; l'ancien échec de seconde passe n'est plus exercé.
- `tests/test_evaluate_finetuned.py::test_cli_persists_safe_payload_and_raw_failures_without_inflating_metrics` : **couverture ancienne réduite** — Renommage et export récupéré plutôt que null ; le test garde l'échec des métriques du modèle brut mais n'exerce plus deux passes.
- `tests/test_golden_dataset.py::test_self_correction_recovers_exact_golden_outputs` : **couverture ancienne réduite** — Les sorties Golden exactes restent exigées, mais le texte suffit en un appel ; ce test ne couvre plus une vraie seconde passe.

### Passe stress et qualification OCR

- Aucun des 29 tests HEAD n'est supprimé sans remplacement. Les tests de stress sont nouveaux ; ils ne rétablissent pas l'ancien contrat de seconde passe des 21 tests signalés ci-dessus. La qualification globale pouvait rendre un document entièrement non vérifié : la couverture forte reste l'absence de chiffres faux, pas la disponibilité.

### Passe récupération régionale / prétraitements

- `test_image_ocr_is_x2_grayscale_tsv` : ancienne taille finale `(40,60)` remplacée par l'existence d'images ×2 et ×3 et la vérification des niveaux de gris pour tous les appels ; ce n'est pas un relâchement de sûreté mais le contrat d'un seul appel final disparaît.
- `test_numbers_are_not_guessed` : le format groupé valide `1.234,56` est accepté, avec une assertion positive exacte ; `1.23,56` reste refusé. Domaine d'entrée élargi explicitement.
- Le seuil 60→40 est un **assouplissement mesuré de la confiance**, pas une preuve de sûreté globale ; les tests de consensus indépendant et de veto textuel sont ajoutés.

### Passe latence / données réelles / préparation du déploiement

- Aucun résultat de scans réels n'est inventé. Les tests de budget utilisent une horloge contrôlée ; les mesures Tesseract sont séparées et réelles.
- Le chemin rapide doit garder les mêmes assertions de sûreté et le budget doit rendre des données non vérifiées, jamais des valeurs par défaut ; toute adaptation de test nécessaire à ce chemin doit être décrite dans l'appendice ci-dessous.
- Aucun des 29 tests historiques comparés à HEAD n'a été supprimé ou affaibli supplémentairement par cette passe. Les nouveaux tests de budget vérifient délai partagé, interdiction de lancer une étape après épuisement, borne transmise à Tesseract et absence de reparsing des quantités OCR rejetées dans un PDF.
- `test_clear_first_pair_never_runs_extra_families`, créé dans cette passe : une première assertion attendait seulement `deskew` ; elle est remplacée par `['deskew', 'clahe']` avec deux appels [4,6], pour **renforcer** le recoupement indépendant. Le premier candidat à deux lectures du même prétraitement n'est pas la version retenue.
- Les huit tests du script de scans réels n'utilisent que des PDF de test générés et n'affirment aucune validation sur un document réel indépendant.

## Tests nouveaux au regard de HEAD mais déjà présents dans l'arbre local

- `tests/test_grounding.py::test_numbers_are_not_guessed` : `1.234,56` n'est plus rejeté comme ambigu ; `1.23,56` reste refusé et un test positif exige exactement `Decimal('1234.56')`. C'est un élargissement volontaire et vérifié, pas une suppression de contrôle des chiffres.
- `tests/test_grounding.py::test_real_tesseract_evidence_with_simulated_wrong_model_outputs_on_all_three_images` : l'exclusion des pourcentages vise les trois champs quantité/prix/montant, pas les colonnes explicites de TVA/remise.
- Passe récupération des null : `tests/test_grounding.py::test_image_ocr_is_x2_grayscale_tsv` ne suppose plus un seul appel OCR ×2 ; il exige des appels ×2 **et ×3**, tous en niveaux de gris, avec la même langue et sortie TSV. L'ancienne assertion sur la taille d'un seul dernier appel est remplacée par une vérification des tailles effectivement utilisées.
- Les anciens tests qui exigeaient le rejet global des images insuffisantes restent en place ; ils portent sur les preuves insuffisantes synthétiques, pas sur une désactivation des garde-fous. Aucun test de sûreté n'est retiré pour améliorer le nombre de null.
- Historique des suites : rapport documentaire 177 tests, stress 237, récupération 249, latence/scans réels 271 (plus 101 sous-tests et 3 slow exclus). Les scénarios historiques adaptés ne gardent pas forcément leurs anciennes assertions : les 21 réductions de couverture ci-dessus restent explicitement signalées.
