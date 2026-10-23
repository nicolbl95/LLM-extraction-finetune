# Alignement Pillow — validation en environnement vierge

## Changement approuvé

```diff
-Pillow<11.0
+Pillow>=11.0,<13.0
```

Gradio 6.30.0 déclare `pillow>=8,<13` et `numpy>=1,<3`. PyMuPDF 2.0 déclare
`numpy` sans borne et aucune dépendance Pillow. OpenCV headless 4.14.0.94
requiert `numpy>=2` sur Python >=3.9 et n'a pas de dépendance Pillow.
Pytesseract 0.3.13 requiert `Pillow>=8` et n'a pas de dépendance NumPy.
La plage du manifeste Pillow 11–12 est donc compatible avec chaque contrainte
métadonnée examinée ; le pin racine NumPy `>=2,<3` l'est aussi.

## Environnement réellement créé

- Python 3.14.2, venv neuf `.venv-pillow-verify` dans le workspace.
- Torch CPU `2.14.1+cpu` préinstallé dans le venv pour éviter de télécharger les
  dépendances CUDA sur une machine sans GPU ; ensuite `pip install -r requirements.txt`
  a été exécuté intégralement.
- Versions installées : Gradio 6.30.0, Pillow 12.3.0, NumPy 2.5.4,
  OpenCV headless 4.14.0.94, PyMuPDF 2.0, pytesseract 0.3.13.
- `pip check` : `No broken requirements found.`
- Tesseract 5.3.4 et les langues `fra`, `eng`, `osd` présents.
- Torch CUDA indisponible. Aucun modèle n'a été chargé ni téléchargé.
- `pytest` n'étant pas déclaré dans les requirements runtime, pytest 9.1.1 a été
  ajouté au venv uniquement comme outil de test.

## Tests frais

- `python -m pytest -q` : **271 passed, 3 deselected, 101 subtests passed**.
- `python scripts/run_stress_check.py --compare results/stress-before-latency.json ...`
  : 35 cas, 612 champs, **0 faux conservé / 0 anomalie fantôme / 0 crash / 155 null**.
- Comparaison : hausse de nulls **0,000 point** (limite +2 points satisfaite).
- 105 profils additionnels : 0 faux conservé, 0 fantôme.

Le venv temporaire a été supprimé après validation. Aucun changement autre que
la ligne Pillow approuvée n'a été apporté au code pendant cette validation.
Aucun commit et aucun push.
