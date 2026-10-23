# Validation sur scans réels — protocole figé, sans réglage

## Constituer l'échantillon avant l'exécution

Réunir **5 à 10 factures anonymisées et autorisées**, indépendantes du
générateur et des exemples du projet : FR/UK/US, scan à plat, photo de téléphone,
PDF natif, photo légèrement inclinée et document multi-page. Garder leurs
difficultés naturelles ; ne pas les choisir après avoir vu les résultats.
Supprimer ou remplacer données personnelles, coordonnées bancaires et clients
identifiables **avant** de remettre les fichiers. Ne rien envoyer à un service tiers.
Le script traite les documents localement et ne charge/télécharge aucun LLM.

Créer un dossier local distinct, par exemple `real-scans/` (ne pas commiter
les factures ni les résultats contenant des données confidentielles). Produire
la vérité terrain **à la main, avant l'outil**, idéalement par deux personnes.
Transcrire littéralement description, fournisseur, date, quantités, prix,
montants HT et TTC imprimés ; `null` si la lecture humaine est indécidable.
Ne pas corriger un calcul imprimé faux : il doit rester fidèle et être annoté
séparément pour la revue des `validation.flags`.

## Format facultatif de vérité terrain

```json
{
  "prepared_before_evaluation": true,
  "documents": {
    "facture-01.pdf": {
      "supplier_name": "Exemple Ltd",
      "invoice_date": "2026-01-01",
      "total_ht": 20,
      "total_ttc": 24,
      "line_items": [
        {"description": "Service", "quantity": 2, "unit_price": 10, "total_ht": 20}
      ]
    }
  }
}
```

Les noms relatifs doivent correspondre **exactement** aux fichiers du dossier.
Les ordres des articles comptent. Conserver une copie horodatée et l'empreinte
du JSON avant exécution ; le programme enregistre son SHA-256, mais ne peut
pas prouver à lui seul que la saisie a précédé l'exécution.

## Exécuter sans adapter les seuils aux résultats

```bash
python scripts/evaluate_real_scans.py real-scans/ \
  --truth real-scans-truth.json --output results/real-scans-review.json
# Sans référence : revue manuelle seulement, aucune exactitude revendiquée.
python scripts/evaluate_real_scans.py real-scans/ \
  --output results/real-scans-manual-review.json
```

Le résultat comprend par document : SHA-256, durée hors GPU, appels Tesseract,
JSON final, champs neutralisés, statuts, anomalies et erreurs techniques.
Il exporte aussi la version et le SHA de la politique OCR unique (`src/ocr_config.py`) :
confiance 40, familles figées et budget total 45 s. Les pages scannées d'un même
PDF partagent ce budget ; son épuisement n'autorise pas de valeurs de repli.
**Sans vérité terrain**, aucun taux d'erreur ni « valeur fausse » n'est annoncé :
les champs neutralisés et drapeaux doivent être relus manuellement. Avec
vérité terrain, les erreurs conservées et `null` à tort sont listés par chemin.
Une valeur fausse conservée fait échouer la commande.

**Interdit après lecture des résultats :** retoucher confiance, familles,
gabarits, budget ou règles sur ces 5–10 factures puis les présenter comme
validation indépendante. Archiver SHA du code, configuration OCR versionnée,
vérité terrain, résultats et limites ; toute modification impose un nouvel
échantillon indépendant. Les drapeaux faux/omis exigent une annotation humaine :
la vérité terrain du JSON seule ne certifie pas la justesse du rapport fiscal.

## Décision avant publication

Bloquant : toute valeur fausse conservée, anomalie fantôme confirmée ou crash.
Rapporter aussi couverture, médiane/p95/pire temps, qualité insuffisante et
documents arrivés au budget. Les 35 stress synthétiques et les trois exemples
ne remplacent pas ce contrôle. Le vrai Qwen se vérifie séparément via
`run_real_model_check.py --stress`, sur une machine explicitement équipée.
**Aucun scan réel indépendant n'a été fourni ni évalué dans cette passe.**
