# PR137 — Validation fonctionnelle et visuelle de la démo recruteur

**Verdict : PASS avec 2 réserves divulguées. 0 ligne de code modifiée. Serveur laissé en marche sur `http://127.0.0.1:7860`.**

- Référence : commit GitHub `97435d7c0670d013c84467ecdadcd4e12d45a8be`
- Branche locale : `wip/layouts-reprise`, HEAD = `97435d7` — identique à `refs/heads/wip/layouts-reprise` sur `origin` (vérifié par `git ls-remote`)
- Session : 2026-11-12, 20 h 55 → 21 h 20 (heures locales Windows)

---

## 1. Phase 1 — préconditions

| Contrôle | Résultat | Preuve |
|---|---|---|
| HEAD / branche | `97435d7c0670d013c84467ecdadcd4e12d45a8be` sur `wip/layouts-reprise` | `git rev-parse` |
| Fichiers de production = version PR136 | **OUI** | `git status --porcelain` ne contient **aucune** entrée sous `src/` ni `app/` |
| Modifications locales préservées | 4 entrées, toutes attendues | ` M packages.txt` (phantome CRLF `autocrlf=true`), ` M scripts/diagnose_independent_benchmark.py`, ` M scripts/experiment_ocr.py`, ` M scripts/shadow_fast_path.py` |
| Python | `.venv/Scripts/python.exe`, Python 3.12 | ligne de commande du processus serveur |
| Gradio | 6.30.0 | `window.gradio_config.version` côté page |
| Tesseract | `5.4.0.20240606`, `langues=eng, fra, osd` | log serveur, ligne `Runtime versions:` |
| Langues OCR effectives | `fra+eng disponibles` | log serveur, ligne `Startup OCR: available=True; fra+eng disponibles; revision=97435d7… (modifications locales non commitées)` |
| Port 7860 | libre avant le redémarrage (0 écouteur, 0 `python.exe`, `curl` exit 7) | Phase 1 ; aucun processus tiers n'a donc eu à être arrêté |
| Qwen/LoRA | **non chargés, aucun modèle téléchargé** | `RuntimeError: CUDA is unavailable…` levé **avant** tout chargement de poids, rattrapé comme récupérable → repli OCR seul documenté |

## 2. Phase 2 — redémarrage et identité du processus

Lanceur établi, copié en nouvelle version pour ne pas écraser le log de la session morte :
`%TEMP%\start-invoice-app-detached-v5.ps1` (identique à v4, seul le chemin du log change) →
`%TEMP%\gradio-7860-pr137.log`.

- Création via WMI (`Invoke-CimMethod Win32_Process Create`) : le parent du lanceur est `WmiPrvSE.exe`, hors de l'arbre du session agent.
- Chaîne de processus vérifiée en fin de session : `21064(python.exe) → 2572(python.exe) → 11168(powershell.exe)`
  avec `pythonTotal=2` — c'est le duo normal venv (redirecteur) → interpréteur de base ; **aucun processus étranger**.
- `INVOICE_DEVICE` reste **non défini** (le forcer à `cpu` déclenche un chargement fp32 ≈ 12 Go, tué par l'OS sur 8 Go).
- HTTP : `curl http://127.0.0.1:7860/` → **200**, 40 845 octets (à la fin de la validation, serveur encore en marche).

## 3. Phase 3 — critères du brief, évalués dans le navigateur

| Critère | Verdict | Observation réelle |
|---|---|---|
| Chargement correct de l'interface | OK | titre `Extraction standardisée de données de factures`, sous-titre, zone d'import `Importer une facture — PDF ou image`, panneau `Résultat de l'extraction — JSON structuré` |
| Affichage des exemples | OK (3/3) | tuiles `France · EUR`, `Royaume-Uni · GBP`, `États-Unis · USD` ; chacune remplace bien le fichier chargé : `france.png 56.0 KB`, `uk.png 63.4 KB`, `us.png 68.9 KB` |
| Possibilité d'importer une facture | OK | upload client réel (`DataTransfer` + événement `change`) de `ma-facture-importee.png`, 57 321 octets, sur un `input[type=file]` avec `accept=".pdf, .png, .jpg, .jpeg"` ; aperçu puis extraction réussis |
| Bouton d'extraction | OK | `Extraire les données` (variant primary) produit un résultat à chaque appui |
| Affichage du JSON structuré | OK | France 4 lignes ; UK `Oak Studio Ltd` 2025-05-18, 3 lignes, 5 912.17 + 447.20 + 168.15 = 6 527.52 = `total_ht` ; US `Cedar Office LLC` 2023-12-07, 5 lignes |
| Présentation du rapport de vérification | OK | France : `⚠️ Anomalies à vérifier — 20/20 champs recoupés` + `Couverture : 4/4 lignes documentées présentes ; exhaustivité de la lecture non établie.` + note SIRET (clé de contrôle) ; UK : `✅ Extraction cohérente — 16/16 champs recoupés` ; US : `✅ Extraction cohérente — 24/24 champs recoupés` |
| Téléchargement du JSON | OK | une extraction écrit `invoice-*.json` dans le cache Gradio ; l'artefact de la dernière exécution relancé depuis la page : **200**, `application/json`, **557 octets**, 5 clés, 3 lignes, contenu identique au panneau affiché |
| Absence de texte technique développeur | OK | balayage des 1 407 caractères visibles sur 17 termes (`Qwen`, `LoRA`, `GPU`, `CUDA`, `Tesseract`, `RapidOCR`, `bitsandbytes`, `torch`, `traceback`, `Exception`, `RuntimeError`, `safetensors`, `adapter`, `OCR seul`, `python`, `venv`, `model_name`) → **0 occurrence** ; console du navigateur : **0 message, 0 erreur** |

### Bloc Performances (exigence littérale du brief)

DOM réellement rendu :

```html
<h3>Performances</h3>
<ul><li><strong>Temps de traitement : 2,62 s</strong></li></ul>
```

Contenu = titre + une seule ligne de durée, sans mention de Qwen, LoRA ni GPU. Durées observées sur la
session : 24,03 s (première extraction, OCR froid), 11,27 s, 10,62 s, 3,74 s (UK), 2,62 s (US).

### Habillage du framework (non du projet)

Visible pour un recruteur dans le pied de page Gradio : `Runs`, `Créé avec Gradio`, `Paramètres`.
`Utiliser via API` et `Stop Recording` sont présents dans le DOM mais **non visibles** (boîte de taille nulle).
Pendant une job en cours, Gradio affiche des minuteurs transitoires (`0.0s`) par composant ; après la fin,
**0 minuteur visible**. Ces textes viennent du bundle Svelte de Gradio, `app/main.py` ne les écrit pas ;
les retirer serait une modification de code/CSS non autorisée en PR137 → **décision propriétaire**.

## 4. Rétractation d'une observation erronée

À mi-session, j'ai relevé que « cliquer `Royaume-Uni · GBP` renvoie le payload France ». Cette observation
est **fausse et due à mon propre outillage** : mes sondages DOM avaient patché `document.createElement` et
`HTMLAnchorElement.prototype.click`, plusieurs scripts avaient été interrompus par la limite de 15 s de
l'outil avant leur ligne de restauration, et mon attente de 2,5 s était plus courte que l'aller-retour
`load_example`. Après rechargement de la page à l'état propre, la tuile `États-Unis · USD` remplace le
fichier chargé en **600 ms** et l'extraction qui suit renvoie bien le document américain. **Aucun défaut
produit** ; le point est clos.

## 5. Réserves et limitations (à lire avant de conclure « validé visuellement »)

1. **Pas de preuve au niveau pixels.** Dans le navigateur intégré, `take_screenshot` et `click` échouent avec
   `NATIVE_BROWSER_VIEWPORT_UNAVAILABLE` (viewport 0×0, caché). Toute la validation visuelle est donc
   **DOM + géométrie calculée** : le contenu, les libellés et la structure sont prouvés ; le rendu CSS
   (couleurs, alignement, densité réelle à l'écran) **n'est pas vérifié**. Un contrôle humain de 30 secondes
   dans un vrai onglet reste nécessaire pour cette dimension.
2. **Téléchargement natif non observable.** Le clic sur `Télécharger le JSON` n'a rien écrit dans
   `~/Downloads` (onglet caché). La vérification s'est faite sur l'artefact servi par le serveur (200,
   `application/json`, contenu identique), ce qui prouve le fichier et son accessibilité, **pas** le
   comportement d'enregistrement du navigateur de l'utilisateur.
3. **Accents mutilés dans le log, pas dans le produit.** Le log affiche `r├®cup├®r├®` pour `récupéré` :
   la redirection `Out-File` du lanceur PowerShell ré-encode une sortie déjà UTF-8 dans la page de code
   console. Le texte affiché au recruteur est correct (`récupéré (corroboré par OCR)` n'apparaît d'ailleurs
   jamais dans l'interface). Artefact d'environnement, non corrigé (aucune modification autorisée).
4. **Écart de formulation, à décider par le propriétaire.** Le sous-titre dit « avec un modèle spécialisé »
   alors que la démo tourne en OCR seul ; le bloc Performances, lui, ne mentionne aucun modèle depuis PR133.
   PR137 interdit de modifier le code : **constaté, non corrigé**.
5. **Aucun test exécuté en PR137** (`0` pytest). Justification : HEAD == version validée, 0 fichier de
   production modifié, et le brief demandait une validation fonctionnelle et visuelle. La dernière porte de
   validation enregistrée pour cet état reste celle de PR135 part 2 (clone propre : 1 091 passed / 0 failed /
   0 errors) et PR134/PR133 (1 168 passed dans l'arborescence).

## 6. Verdict

`PASS_WITH_DISCLOSED_GAPS` — les 8 critères du brief Phase 3 sont satisfaits sur le DOM de la démo réelle,
le bloc Performances est exactement conforme à l'exigence, et l'interface ne montre aucun texte technique
destiné aux développeurs. Les deux réserves (rendu pixels non prouvé, téléchargement natif non observable)
sont des limitations de l'outillage de test, pas des défauts du produit. Le pied de page Gradio et le
sous-titre « modèle spécialisé » sont deux points qui relèvent d'une décision propriétaire, pas d'un correctif.

## 7. Handover

1. **État** : `wip/layouts-reprise` @ `97435d7` (= tip publié `origin`), 4 modifications locales préservées
   (`packages.txt` + 3 scripts de recherche), 0 fichier `src/`/`app/` modifié, 0 commit créé, 0 push.
   Démo OCR seule en marche sur `http://127.0.0.1:7860` (PID serveur 21064, log `gradio-7860-pr137.log`).
2. **Actions** : préconditions ; redémarrage via le lanceur WMI v5 avec preuve de chaîne de processus ;
   validation navigateur des 8 critères (3 exemples, import réel, extraction, JSON, rapport, téléchargement,
   Performances, scan de texte technique) ; rétractation documentée d'une fausse observation.
3. **Fichiers** : ce rapport et `pr137-live-verification.json` sous `results/product-review137/` ;
   `%TEMP%\start-invoice-app-detached-v5.ps1` (copie de v4, chemin de log uniquement) et
   `%TEMP%\pr137-chain.ps1` (preuve de chaîne) hors dépôt. **Aucun fichier de production touché.**
4. **Commandes / résultats** : `git rev-parse`/`git ls-remote` (HEAD = tip serveur) ; `curl` → 200 /
   40 845 octets ; lecture du log (`fra+eng disponibles`, `revision=97435d7…`) ; `powershell -File
   pr137-chain.ps1` → chaîne `21064 → 2572 → 11168`, `pythonTotal=2` ; `ls Temp/gradio/invoice-*.json` ;
   fetch same-origin de l'artefact JSON → 200 / `application/json` / 557 octets ; **0 suite de tests lancée**.
5. **Blockers** : aucun bloquant. Limitations d'outillage listées en §5 (pas de capture pixel ; pas de
   téléchargement natif observable dans l'onglet caché).
6. **Prochaines étapes recommandées** : (a) laisser l'utilisateur ouvrir l'onglet 30 s pour la validation
   pixels ; (b) décision propriétaire sur le pied de page Gradio et le sous-titre « modèle spécialisé » ;
   (c) ne rien publier sur Hugging Face ; (d) ne pas enchaîner automatiquement une nouvelle phase.
7. **Périmètre** : PR137 s'arrête ici, conformément au brief (« Cette phase est une validation fonctionnelle
   et visuelle ») ; aucune phase suivante n'est démarrée.
