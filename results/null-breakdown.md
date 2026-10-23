# Diagnostic des null à tort — référence figée avant réglage

**416 null sur 612 champs attendus, 35 cas.** Seuils inchangés au moment du diagnostic : moyenne 85/100, token 60/100.

Les **600 champs** correspondaient aux 34 cas antérieurs ; l'ajout de la remise monétaire séparée ajoute 12 champs. Le jeu fixe de 35 cas comporte **612 champs**.

| Famille | Cas | Champs attendus | Null à tort |
| --- | ---: | ---: | ---: |
| france | 8 | 160 | 160 |
| uk | 8 | 128 | 112 |
| us | 8 | 192 | 144 |
| layout | 11 | 132 | 0 |

## Cause principale exclusive

Chaque null hérite de la première cause bloquante, dans l'ordre 1 → 6 ci-dessous. Il s'agit de l'attribution du refus du document entier, **pas** de la preuve que chaque champ était illisible. Les causes se recouvrent ; les messages bruts restent dans le JSON.

| Cause | Null attribués |
| --- | ---: |
| 1. Aucun token OCR exploitable | 60 |
| 2. Confiance sous le seuil | 60 |
| 3. Segmentations en désaccord | 64 |
| 4. Tableau/ligne non reconstruite | 200 |
| 5. Contrainte sans solution unique | 0 |
| 6. Autre / garde-fou documentaire | 32 |

### Trois causes principales

- **4. Tableau/ligne non reconstruite : 200 null**.
- **3. Segmentations en désaccord : 64 null**.
- **2. Confiance sous le seuil : 60 null**.

## Par dégradation et champ

| Dégradation | invoice_date | lignes.description | lignes.quantity | lignes.total_ht | lignes.unit_price | supplier_name | total_ht | total_ttc | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 3 | 12 | 12 | 12 | 12 | 3 | 3 | 3 | 60 |
| destroyed | 3 | 12 | 12 | 12 | 12 | 3 | 3 | 3 | 60 |
| half_resolution | 3 | 12 | 12 | 12 | 12 | 3 | 3 | 3 | 60 |
| jpeg40 | 2 | 7 | 7 | 7 | 7 | 2 | 2 | 2 | 36 |
| low_contrast | 1 | 4 | 4 | 4 | 4 | 1 | 1 | 1 | 20 |
| noise | 3 | 12 | 12 | 12 | 12 | 3 | 3 | 3 | 60 |
| rotation | 6 | 24 | 24 | 24 | 24 | 6 | 6 | 6 | 120 |

| Cause | invoice_date | lignes.description | lignes.quantity | lignes.total_ht | lignes.unit_price | supplier_name | total_ht | total_ttc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. Aucun token OCR exploitable | 3 | 12 | 12 | 12 | 12 | 3 | 3 | 3 |
| 2. Confiance sous le seuil | 3 | 12 | 12 | 12 | 12 | 3 | 3 | 3 |
| 3. Segmentations en désaccord | 3 | 13 | 13 | 13 | 13 | 3 | 3 | 3 |
| 4. Tableau/ligne non reconstruite | 10 | 40 | 40 | 40 | 40 | 10 | 10 | 10 |
| 5. Contrainte sans solution unique | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6. Autre / garde-fou documentaire | 2 | 6 | 6 | 6 | 6 | 2 | 2 | 2 |

## Décision préalable

Ne pas baisser les seuils : beaucoup de valeurs justes sont annulées par un garde-fou **global**, surtout lorsque le tableau n'est pas reconstruit. Priorité au redressement et au recoupement par champ ; mesurer les prétraitements sans utiliser les références comme source de valeurs. Une courbe de seuil sera descriptive, et le choix ne pourra utiliser que la partition de réglage.
