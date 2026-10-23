# Publication du rapport documentaire — plan à valider

**Aucune publication autorisée dans cette passe.** Sauvegarde locale créée :
`backup/pre-document-report-137b643` → `137b643`. Ce tag n'a pas été poussé ;
il protège le dernier commit, **pas** l'arbre local non commité.

## 1. Avant de commiter

- Examiner `git diff` et les nouveaux fichiers, puis `docs/TEST_CHANGES.md`
  (contrats historiques retirés explicitement signalés).
- Inclure : `app/main.py`, `src/evaluate_finetuned.py`,
  `src/invoice_{evidence,grounding,validation}.py`, `src/ocr_quality.py`,
  `src/evaluate_grounding.py`, `src/diagnose_pipeline.py`,
  `src/ocr_{preprocessing,consensus,recovery,config,budget}.py`, `scripts/*.py`,
  `tests/stress/`, `tests/fixtures/`, nouveaux tests et adaptations existantes,
  `pytest.ini`, `requirements.txt`, `requirements-inference.txt`,
  `notebooks/real_model_check.ipynb`, `README.md`, cette documentation.
- Ne pas ajouter poids, images temporaires dégradées, caches, logs ou anciens
  patchs de travail. Les rapports de stress et rendus sont des livrables de revue,
  pas des dépendances d'exécution. Ne jamais inclure de token.
- Exécuter `python -m pytest -q`, `git diff --check`,
  `python scripts/run_stress_check.py --output results/stress-report.json`.
  Lire les **cas**, pas seulement le total : accepter les `null` ne garantit
  pas une bonne couverture. Le vrai modèle reste un verrou séparé.

## 2. Dépendances et préproduction

- Space : SDK Gradio **6.30.0** dans le YAML du README, pas de seconde version
  contradictoire dans `requirements.txt`. Versions observées et contraintes :
  Pydantic `>=2,<3`, Transformers `>=4.49,<6`, PEFT `>=0.14`, Torch `>=2.5`,
  Pillow `>=11,<13`, NumPy `>=2,<3`, Hub sans ancien plafond. Ce ne sont pas des dépendances verrouillées :
  capturer `pip freeze` sur la machine de préproduction et conserver les logs.
  README ne fixe pas `python_version` ; Spaces utilise Python **3.10 par défaut**
  selon la Configuration Reference. Gradio 6.30.0 déclare Python `>=3.10`.
- `packages.txt` doit contenir `tesseract-ocr`, `tesseract-ocr-fra`,
  `tesseract-ocr-eng`. Vérifier `tesseract --list-langs` et
  `python -c "from app.main import ocr_availability; print(ocr_availability())"`.
- Adaptateurs et **modèle de base** doivent être disponibles ; vérifier CUDA,
  bitsandbytes, RAM GPU et tokenizer. Sur GPU/Colab, lancer la commande réelle
  du README : trois répétitions par image, aucune réussite OCR seule acceptée.
- Prétraitements : `opencv-python-headless>=4.10,<5` et NumPy via dépendances ;
  NumPy est maintenant déclaré explicitement `numpy>=2,<3`. La roue headless
  locale indique `GUI: NONE` : aucun paquet Qt/X11 ne manque à cette lecture OCR.
  `packages.txt` couvre Tesseract et `fra+eng` (5.3.4 constaté localement).
- Le solveur a été exécuté dans un vrai Python **3.10.22** (dry-run sans
  installation) : il résout le manifeste avec Gradio 6.30.0, Pillow 12.3.0,
  NumPy 2.2.6 et PyMuPDF 1.28.2. Cela confirme la compatibilité du solveur,
  pas un build ni une exécution sur le Space. Le venv fraîchement installé avec
  le manifeste a utilisé Python 3.14.2/Pillow 12.3.0 ; ces deux essais sont distincts.
- Autres versions locales observées : NumPy 2.5.4, OpenCV headless 4.14.0.94,
  PyMuPDF 2.0, Pydantic 2.14.1, Hub 2.2.0. Torch, Transformers, PEFT et
  bitsandbytes sont **absents localement** : ne pas présenter cette vérification
  OCR comme une installation complète d'inférence validée sur HF.
- Obtenir l'accord explicite avant commit/push ; créer un commit cohérent, puis
  publier normalement. Ne pas forcer la réécriture de `main`.

## 3. Vérifications après publication autorisée

1. Enregistrer le SHA publié, le comparer au **hash en pied de page**.
   Une mention d'arbre local modifié n'est pas un déploiement certifié.
2. Pied de page : **« OCR : disponible »** ; logs de démarrage `fra+eng`.
3. Téléverser les trois exemples : FR HT/TTC 5716,77/6860,12 ;
   UK 6527,52/7743,58 ; US 8972,80/8972,80. US ligne 3 : 5,1/77,66/396,07.
4. Vérifier le rapport avant JSON, détails fermés, pas d'erreur de facture
   causée par une sortie modèle ; vraie erreur imprimée dans `validation.flags`.
5. Tester une image dégradée : « Preuves OCR insuffisantes », valeurs non
   recoupées `null`, statuts non vérifiés. Vérifier le chemin propre **2 appels**,
   l'escalade jusqu'à **12 appels**, et le budget **45 s partagé par document**.
   L'arrêt du processus Tesseract et le rendu ajoutent une petite surcharge
   mesurée ; ne pas annoncer un hard timeout absolu de 45,000 s.
6. Relever quota, appels, réponses brutes et durée ; ne pas interpréter un repli
   OCR comme une génération réelle. Répéter trois fois sur le même matériel.
7. Relever version et SHA de `src/ocr_config.py`, message de limite connue en
   pied de page, et suivre `docs/REAL_SCANS.md` sur 5–10 documents nouveaux
   anonymisés avant tout réglage. Les mesures actuelles ne sont pas des scans
   réels indépendants et ne certifient pas la latence du Space.

## 4. Retour arrière

Conserver le SHA de publication et le tag de sauvegarde. En cas de régression :
`git revert <SHA-du-commit-publié>` (ou `git revert -m 1 <SHA-merge>` pour un merge),
tester, puis pousser **le revert**, après accord, sans `reset --hard` ni force-push.
Pour une branche de diagnostic : `git switch -c rollback-check backup/pre-document-report-137b643`
uniquement depuis un arbre propre/sauvegardé. Le tag précédent ne prouve pas que
l'ancien comportement était sûr : restaurer l'application n'autorise pas à
conserver des chiffres non ancrés. Refaire les contrôles de hash/OCR et des exemples.
