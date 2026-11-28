import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Usine 4.0 - Brique à nommer",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Le dossier outils/ est à la racine du dépôt, un niveau au-dessus de projet/.
    sys.path.insert(0, str(Path(mo.notebook_dir()).parent / "outils"))
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return mo, np, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projet Usine 4.0 — Brique 2 : Localisation

    **Groupe :** B · **Membres :** Loric HAGARD, Abdellah DJEDIDI, Florianne PAN, Junjie (Eric) SHAO

    **Mode d'emploi de ce modèle**

    - Copiez ce fichier dans `projet/` (par exemple sous le nom indiqué à la fin du TD,
      `brique1_classification.py` ; le sujet est dans `projet/SUJET.md`),
      puis ouvrez la copie : `marimo edit projet/brique1_classification.py`.
    - Les fonctions fournies restent accessibles par `tp.` (chargement des données, affichage, cache).
      Les fonctions que vous avez écrites dans un TD ne s'importent pas d'un notebook à l'autre : copiez dans
      une cellule de cette brique celles dont vous avez besoin.
    - Dans une cellule, une variable dont le nom commence par un tiret bas reste locale à la cellule.
    - Fixez les graines (`tp.fixer_graine(0)`) et gardez les résultats des calculs longs
      (`tp.sauver_cache`, `tp.charger_cache`) pour ne pas les relancer à chaque ouverture.
    - Ne réglez rien sur le jeu de test : gardez une partie des données pour la validation, et ne mesurez sur
      le test qu'une fois vos choix faits.
    - Remplacez les `...` par votre travail et supprimez ce mode d'emploi avant le rendu.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. La question et les données

    Quelle question métier cette brique traite-t-elle ? Quelles données, en quelle quantité, comment
    sont-elles découpées (entraînement, validation, test) ?

    Cette brique traite de la localisation automatique des défauts de surface sur des bandes d'acier laminées à chaud, afin d'isoler ou de router les zones non conformes sans interrompre la chaîne de production.

    On utilise comme données la base NEU-DET contenant 6 types de défauts de surface (*crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches*). Il y a 1 800 images au total (300 images par classe). Le découpage est aligné sur celui de la Brique 1 :On les découpe tel que:
    Entraînement : 1 080 images (180 par classe), Validation : 120 images (20 par classes), Test : 600 images (100 par classe)
    """)
    return


@app.cell
def _(mo, np, tp):
    # Chargement et découpage des données. Exemple pour les images NEU (voir outils/tp_utils.py) :
    tp.fixer_graine(0)
    _cible = tp.DONNEES / "neu-yolo-projet"
    config_neu = tp.voc_vers_yolo(tp.DONNEES / "NEU", _cible, tp.NEU_CLASSES, parties=tp.decoupage_neu())
    images_val, images_test = tp.lister_images(_cible, "val"), tp.lister_images(_cible, "test")

    _images_train = tp.lister_images(_cible, "train")
    _rng = np.random.default_rng(0)
    _choix = [_images_train[i] for i in _rng.choice(len(_images_train), 6, replace=False)]
    _imgs = []
    for _p in _choix:
        _b, _c = tp.lire_annotations(_p)
        _imgs.append(tp.dessiner_boites(tp.Image.open(_p), _b, _c, tp.NEU_CLASSES))

    mo.vstack([
        mo.md(f"**{len(_images_train)}** train, **{len(images_val)}** val, **{len(images_test)}** test."),
        tp.grille(_imgs, n_col=3, taille=3.0)
    ])
    return config_neu, images_test, images_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. La baseline classique

    Une méthode sans réseau de neurones, raisonnable pour ce problème : laquelle, pourquoi, quel score ?

    ...
    """)
    return


@app.cell
def _(images_test, images_val, mo, np, tp):
    import cv2

    # Cellule 1 : Fonctions utilitaires (IoU, extraction des boîtes et évaluation)
    def iou(boite_a, boite_b):
        aire_a = (boite_a[2] - boite_a[0]) * (boite_a[3] - boite_a[1])
        aire_b = (boite_b[2] - boite_b[0]) * (boite_b[3] - boite_b[1])
        i_x1, i_y1 = max(boite_a[0], boite_b[0]), max(boite_a[1], boite_b[1])
        i_x2, i_y2 = min(boite_a[2], boite_b[2]), min(boite_a[3], boite_b[3])
        aire_inter = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
        aire_union = aire_a + aire_b - aire_inter
        return aire_inter / aire_union if aire_union > 0 else 0.0

    def boites_depuis_masque(masque, aire_min):
        n, _, stats, _ = cv2.connectedComponentsWithStats(masque.astype(np.uint8))
        return [
            [stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]]
            for i in range(1, n)
            if stats[i][4] >= aire_min
        ]

    def evaluer_detections(predictions, verites, seuil_iou=0.3):
        preds = sorted(predictions, key=lambda x: x[1], reverse=True)
        tp, fp = 0, 0
        verites_prises = set()
        for boite_pred, _ in preds:
            meilleure_iou, meilleur_id = 0.0, -1
            for i, v in enumerate(verites):
                if i not in verites_prises:
                    val = iou(boite_pred, v)
                    if val >= meilleure_iou:
                        meilleure_iou, meilleur_id = val, i
            if meilleur_id != -1 and meilleure_iou >= seuil_iou:
                tp += 1
                verites_prises.add(meilleur_id)
            else:
                fp += 1
        fn = len(verites) - len(verites_prises)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rappel = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "rappel": rappel}

        import cv2 
    # Même mesure pour les deux méthodes : une boîte est correcte si son IoU avec un défaut annoté atteint 0,3,
    # quel que soit le type de défaut (la baseline ne sait pas le nommer).
    def compter(images, predictions):
        """Additionne vrais positifs, fausses alertes et oublis sur une liste d'images."""
        total = [0, 0, 0]
        for chemin, boites_scores in zip(images, predictions):
            res = evaluer_detections(boites_scores, list(tp.lire_annotations(chemin)[0]), seuil_iou=0.3)
            for k, cle in enumerate(("tp", "fp", "fn")):
                total[k] += res[cle]
        return total

    def garder(brutes, seuil):
        """Ne garde que les boîtes de YOLO dont le score atteint le seuil."""
        return [[(b, s) for b, s in zip(bx, sc) if s >= seuil] for bx, sc, _ in brutes]

    # Votre baseline :
    def baseline(images, sigma=25, k=2.0, aire=150, noyau=9):
        """Écart au fond local (flou gaussien d'écart-type sigma, en pixels), seuil à moyenne + k écarts-types,
        fermeture (dilatation puis érosion : bouche les petits trous, relie les morceaux proches), composantes."""
        sorties = []
        for chemin in images:
            gris = cv2.cvtColor(tp.lire_image_rgb(chemin), cv2.COLOR_RGB2GRAY).astype(np.float32)
            ecart = np.abs(gris - cv2.GaussianBlur(gris, (0, 0), sigma))
            masque = cv2.morphologyEx((ecart > ecart.mean() + k * ecart.std()).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((noyau, noyau), np.uint8))
            sorties.append([(b, 1.0) for b in boites_depuis_masque(masque, aire)])
        return sorties

    def f1(c):
        return 2 * c[0] / max(1, 2 * c[0] + c[1] + c[2])

    _cache_base = tp.charger_cache("brique2_baseline_neu")

    if _cache_base is not None:
        reglage_baseline = tuple(int(x) if i != 1 else float(x) for i, x in enumerate(_cache_base["reglage"]))
        c_baseline = [int(x) for x in _cache_base["comptage"]]
    else:
        _grille = [
            (s, k, a, n)
            for s in (5, 15, 25, 35)
            for k in (1.5, 2.0, 2.5, 3.0)
            for a in (50, 150, 400)
            for n in (5, 9, 15)
        ]
        reglage_baseline = max(_grille, key=lambda r: f1(compter(images_val, baseline(images_val, *r))))
        c_baseline = compter(images_test, baseline(images_test, *reglage_baseline))
        tp.sauver_cache(
            "brique2_baseline_neu",
            reglage=np.array(reglage_baseline, dtype=float),
            comptage=np.array(c_baseline, dtype=int),
        )

    prec_base = c_baseline[0] / max(1, c_baseline[0] + c_baseline[1])
    rap_base = c_baseline[0] / max(1, c_baseline[0] + c_baseline[2])
    score_baseline = round(f1(c_baseline), 3)

    mo.md(f"""
    **Baseline classique (écart au fond local)** :
    - Meilleur réglage sur validation : flou `sigma={reglage_baseline[0]}`, seuil `k={reglage_baseline[1]}`, aire min `{reglage_baseline[2]} px`, fermeture `{reglage_baseline[3]}×{reglage_baseline[3]}`
    - Résultats sur test ({len(images_test)} images) : Précision **{prec_base:.1%}**, Rappel **{rap_base:.1%}**, F1 **{score_baseline}**

    C'est mieux qu'à la fin du TP2 où on a une précision de 38.3% et un rappel de 38.7% pour la baseline.

    **Limites de la baseline :** Cette approche purement photométrique atteint vite un plafond (F1 autour de 0,38–0,42). Comme elle se base uniquement sur un contraste local par rapport au fond flouté, elle confond les défauts diffus à faible contraste (comme le *crazing* ou les *pitted_surface*) avec le grain naturel du laminage ou des variations d'éclairage, et reste incapable de classifier le type de défaut.
    """)
    return compter, f1, garder, score_baseline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. L'approche par deep learning

    Quel modèle, quelle stratégie d'adaptation, quels réglages essayés (comparés sur la validation) ?

    ...
    """)
    return


@app.cell
def _(compter, config_neu, f1, garder, images_test, images_val, mo, np, tp):
    _dossier_yolo = tp.SORTIES / "yolo" / "neu_projet"
    _poids_yolo = _dossier_yolo / "weights" / "best.pt"

    if not _poids_yolo.exists():
        _m = tp.charger_yolo("yolov8n.pt")
        _m.train(
            data=str(config_neu),epochs=30,imgsz=320, batch=16, workers=0, plots=False, amp=False,
            seed=0, project=str(_dossier_yolo.parent), name="neu_projet", exist_ok=True, verbose=False,
        )

    modele_yolo = tp.charger_yolo(str(_poids_yolo))
    res_yolo_test = modele_yolo.val(data=str(config_neu), split="test", imgsz=320, workers=0, plots=False, verbose=False)
    map_par_classe = dict(zip(tp.NEU_CLASSES, [round(float(m), 3) for m in res_yolo_test.box.maps]))


    _cache_eval = tp.charger_cache("brique2_yolo_eval")
    if _cache_eval is not None:
        seuil_opt = float(_cache_eval["seuil"])
        c_yolo = [int(x) for x in _cache_eval["comptage"]]
    else:
        _brutes_val = tp.predictions_yolo(modele_yolo, images_val, conf=0.01, taille=320, agnostique=True)
        _brutes_test = tp.predictions_yolo(modele_yolo, images_test, conf=0.01, taille=320, agnostique=True)
        seuil_opt = max([round(0.05 * k, 2) for k in range(1, 20)], key=lambda s: f1(compter(images_val, garder(_brutes_val, s))))
        c_yolo = compter(images_test, garder(_brutes_test, seuil_opt))
        tp.sauver_cache("brique2_yolo_eval", seuil=np.array(seuil_opt), comptage=np.array(c_yolo, dtype=int))

    prec_yolo = c_yolo[0] / max(1, c_yolo[0] + c_yolo[1])
    rap_yolo = c_yolo[0] / max(1, c_yolo[0] + c_yolo[2])
    score_modele = round(f1(c_yolo), 3)
    map50_yolo = round(float(res_yolo_test.box.map50), 3)

    mo.md(f"""
    **YOLOv8n (30 époques, 320 px)** :
    - Seuil optimal (choisi sur val) : **`{seuil_opt:.2f}`**
    - Test ({len(images_test)} images) : Précision **{prec_yolo:.1%}** | Rappel **{rap_yolo:.1%}** | **F1 (`score_modele`) : `{score_modele}`** | mAP@0.5 : **`{map50_yolo}`**
    - mAP@0.5 par classe : `{map_par_classe}`
    """)
    return modele_yolo, score_modele, seuil_opt


@app.cell
def _(images_test, mo, modele_yolo, np, seuil_opt, tp):
    _sam = tp.charger_sam()
    _exemples_sam = []
    for _chemin in images_test[:4]:
        _img_rgb = tp.lire_image_rgb(_chemin)
        _boites, _, _ = tp.predictions_yolo(modele_yolo, [_chemin], conf=seuil_opt, taille=320)[0]
        _visu = _img_rgb.copy()
        if len(_boites) > 0:
            _res_sam = _sam(_img_rgb, bboxes=np.array(_boites), verbose=False)[0]
            if _res_sam.masks is not None:
                for _m in _res_sam.masks.data.cpu().numpy():
                    _visu[_m > 0] = (_visu[_m > 0] * 0.5 + np.array([255, 60, 60]) * 0.5).astype(np.uint8)
        _exemples_sam.append(tp.dessiner_boites(tp.Image.fromarray(_visu), _boites))

    mo.vstack([
        tp.grille(_exemples_sam, n_col=4, taille=3.0),
        mo.md("MobileSAM délimite bien les défauts compacts et contrastés (*patches*, *inclusion*), mais son masque déborde sur le fond texturé pour les défauts diffus (*crazing*, *pitted_surface*). La boîte englobante de YOLO suffit donc pour le flux principal.")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    C'est quand même mieux qu'à la fin du TP2 où on a une précision de 66.2% et un rappel de 65.2% pour le modèle YOLO avec seulement 3 époques. On monte a une précision de 72%, un rappel de 78.5% et un f1 de 0.753 avec 30 époques.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Résultats et analyse

    Tableau comparatif (mêmes données, même métrique pour toutes les méthodes), cas d'échec, seuil ou réglage
    retenu et sa justification en termes de coût métier.

    ...
    """)
    return


@app.cell
def _(mo, score_baseline, score_modele, tp):
    # Une ligne par méthode, la même métrique pour toutes ; ajoutez ensuite vos figures et vos cas d'échec.
    tableau_final = [
        {"méthode": "baseline classique", "score sur le test": score_baseline},
        {"méthode": "deep learning", "score sur le test": score_modele},
    ]
    tp.tableau(tableau_final, mo)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Fiche de décision

    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | Pouvoir localiser des défauts sur des pièces d'usine |
    | Baseline classique et son score | écart au fond local, score: F1 = 0.394 |
    | Modèle retenu et pourquoi | Yolo8 fentrainné sur 30 époques très bon pour localiser des objets sur une image, ici les défauts des pièces |
    | Métrique principale et score obtenu | métrique principale : f1 score = 0.753 |
    | Seuil ou réglage retenu, justification métier | seuil choisi : 0.25 ce seuil permet de conserver une bonne précison et un bon rappel et ainsi conserver un bon équilibre entre le coût lorsque la machine ne localise pas correctement un défaut et le coût humain lorsque la machine détecte un défaut qui n'est pas réellement présent. |
    | Coût (annotation, entraînement, inférence, matériel) | ... |
    | Principaux cas d'échec | Les défauts diffus (crazing,pitted_surface) |
    | Le deep learning est-il justifié ici ? | Ici le deep learning est justifié car son score est bien supérieur à celui de la baseline |
    | Déployable en l'état ? Sinon, que manque-t-il ? | Non le modèle mériterait un meilleur entraînement notamment pour les défauts diffus qu'il a encore du mal à repérer |
    """)
    return


if __name__ == "__main__":
    app.run()
