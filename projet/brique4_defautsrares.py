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
    return mo, np, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projet Usine 4.0 — Brique 4  : Défauts rares

    **Groupe :** ... · **Membres :** ...

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

    En usine, certains défauts n'apparaissent que très rarement sur la ligne de production. On n'a donc presque pas d'images annotées pour entraîner un modèle classique dessus. Ici on suit la **Piste A** : construire un petit moteur de recherche par similarité visuelle. Quand un opérateur photographie un défaut rare, le système lui affiche les 5 cas les plus proches dans l'historique pour l'aider à poser un diagnostic.

    On utilise le même découpage officiel NEU-DET que dans les briques 1 et 2 (180 images d'entraînement, 20 de validation et 100 de test par classe), et on réduit la classe rare `scratches` à **15 images** uniquement dans l'historique d'entraînement.
    """)
    return


@app.cell
def _(mo, np, tp):
    # Chargement et découpage des données. Exemple pour les images NEU (voir outils/tp_utils.py) :
    # (images_train, y_train, images_val, y_val, images_test, y_test,
    #  noms) = tp.charger_neu(n_train_par_classe=30, n_val_par_classe=20, n_test_par_classe=100)
    CLASSE_RARE = "scratches"
    N_RARE_TRAIN = 15

    tp.fixer_graine(0)
    parties = tp.decoupage_neu(n_test_par_classe=100, n_val_par_classe=20, graine=0)
    (images_train_complet, y_train_complet, images_val, y_val, images_test, y_test, noms) = tp.charger_neu(
        n_train_par_classe=180, n_val_par_classe=20, n_test_par_classe=100, graine=0
    )

    idx_classe_rare = noms.index(CLASSE_RARE)

    # Réduction de la classe rare à 15 images uniquement dans l'historique d'entraînement
    _rng = np.random.default_rng(0)
    indices_gardes = []
    for _k in range(len(noms)):
        _idx_k = np.flatnonzero(np.asarray(y_train_complet) == _k)
        if _k == idx_classe_rare:
            _idx_k = _rng.choice(_idx_k, size=N_RARE_TRAIN, replace=False)
        indices_gardes.append(_idx_k)
    indices_gardes = np.sort(np.concatenate(indices_gardes))

    images_train = [images_train_complet[i] for i in indices_gardes]
    y_train = np.asarray(y_train_complet)[indices_gardes]

    mo.md(
        f"- **Classe rare choisie** : `{CLASSE_RARE}` (**{N_RARE_TRAIN}** images dans l'historique, contre **180** pour les 5 autres classes)  \n"
        f"- **Historique total (train)** : {len(images_train)} images (au lieu de {len(images_train_complet)})  \n"
        f"- **Validation** : {len(images_val)} images (20/classe) · **Test** : {len(images_test)} images (100/classe)"
    )
    return (
        CLASSE_RARE,
        idx_classe_rare,
        images_test,
        images_train,
        images_train_complet,
        images_val,
        indices_gardes,
        noms,
        y_test,
        y_train,
        y_train_complet,
        y_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. La baseline classique

    Une méthode sans réseau de neurones, raisonnable pour ce problème : laquelle, pourquoi, quel score ?

    on reprend la méthode de l'étape 3 du TP4 : on décrit chaque image avec `tp.descripteur_pixels` (image réduite en 16×16 + histogrammes de niveaux de gris) et on cherche les 5 images les plus proches avec la **similarité cosinus**. Pour mesurer si la recherche est bonne, on calcule la **Précision à 5 (P@5)**, c'est-à-dire la proportion d'images parmi les 5 remontées qui ont vraiment le même défaut que l'image requête.
    """)
    return


@app.cell
def _(
    CLASSE_RARE,
    images_test,
    images_train_complet,
    images_val,
    indices_gardes,
    mo,
    noms,
    np,
    tp,
    y_test,
    y_train,
    y_val,
):
    # Votre baseline : remplacez cette ligne par votre code ; réglez-la sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    def similarite_cosinus(A, B):
        """A : (n, d), B : (m, d) -> tableau (n, m) des similarités cosinus."""
        norme_A = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        norme_B = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
        return (A / norme_A) @ (B / norme_B).T

    def rechercher(emb_requete, emb_corpus, k=5):
        """Indices des k éléments du corpus les plus proches de la requête (similarité cosinus décroissante)."""
        sim = similarite_cosinus(emb_requete[None, :], emb_corpus)[0]
        return np.argsort(-sim)[:k]

    def precision_a_k(indices, pertinents, k=5):
        """Proportion d'images pertinentes parmi les k premiers résultats."""
        return float(np.mean([pertinents[i] for i in indices[:k]]))

    def evaluer_recherche(emb_req, y_req, emb_hist, y_hist, noms_classes, k=5):
        y_req = np.asarray(y_req)
        y_hist = np.asarray(y_hist)
        par_classe = {}
        for c_idx, nom in enumerate(noms_classes):
            sel = np.flatnonzero(y_req == c_idx)
            pert = y_hist == c_idx
            scores = [precision_a_k(rechercher(emb_req[i], emb_hist, k), pert, k) for i in sel]
            par_classe[nom] = round(float(np.mean(scores)), 3)
        return round(float(np.mean(list(par_classe.values()))), 3), par_classe

    _cache_class = tp.charger_cache("brique4_emb_classique")
    if _cache_class is None:
        _enc_c = lambda imgs: np.stack([tp.descripteur_pixels(tp.Image.open(p)) for p in imgs])
        _c_tr = _enc_c(images_train_complet)
        _c_va = _enc_c(images_val)
        _c_te = _enc_c(images_test)
        tp.sauver_cache("brique4_emb_classique", tr=_c_tr, va=_c_va, te=_c_te)
    else:
        _c_tr, _c_va, _c_te = _cache_class["tr"], _cache_class["va"], _cache_class["te"]

    emb_class_train_complet = _c_tr
    emb_class_train = _c_tr[indices_gardes]
    emb_class_val = _c_va
    emb_class_test = _c_te

    _val_class_glob, _val_class_c = evaluer_recherche(emb_class_val, y_val, emb_class_train, y_train, noms, k=5)
    score_baseline, test_class_c = evaluer_recherche(emb_class_test, y_test, emb_class_train, y_train, noms, k=5)

    mo.md(
        f"Baseline classique (pixels + histogrammes + similarité cosinus) :  \n"
        f"- **Validation (P@5 globale)** = **{_val_class_glob:.3f}** (sur `{CLASSE_RARE}` : **{_val_class_c[CLASSE_RARE]:.3f}**)  \n"
        f"- **Test final (P@5 globale)** = **{score_baseline:.3f}** (sur `{CLASSE_RARE}` : **{test_class_c[CLASSE_RARE]:.3f}**)"
    )
    return (
        emb_class_test,
        emb_class_train,
        emb_class_train_complet,
        emb_class_val,
        evaluer_recherche,
        rechercher,
        score_baseline,
        test_class_c,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. L'approche par deep learning

    Quel modèle, quelle stratégie d'adaptation, quels réglages essayés (comparés sur la validation) ?

    Avec seulement 15 exemples pour la classe rare, réentraîner un réseau complet risquerait de faire du surapprentissage. On utilise donc deux *foundation models* déjà pré-entraînés, sans toucher à leurs poids, uniquement pour extraire des embeddings :
    - **CLIP (ViT-B/32)**, qui a été entraîné sur des paires image-texte du web ;
    - **DINOv2**, qui a été entraîné uniquement sur des images sans texte, ce qui l'aide souvent à mieux capter les textures et les formes fines.

    On compare les deux sur le jeu de **validation** (avec toutes les images puis avec la classe rare réduite à 15) pour choisir lequel on garde avant de toucher au test.
    """)
    return


@app.cell
def _(
    CLASSE_RARE,
    emb_class_train,
    emb_class_train_complet,
    emb_class_val,
    evaluer_recherche,
    images_test,
    images_train_complet,
    images_val,
    indices_gardes,
    mo,
    noms,
    np,
    torch,
    tp,
    y_train,
    y_train_complet,
    y_val,
):
    # Votre modèle : remplacez cette ligne par votre code ; réglez-le sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    _cache_clip = tp.charger_cache("brique4_emb_clip")
    if _cache_clip is None:
        _m_clip, _prep_clip, _ = tp.charger_clip()
        _enc_clip = lambda imgs, titre: tp.encoder_images(
            _m_clip, _prep_clip, imgs, progression=lambda it: mo.status.progress_bar(it, title=titre)
        )
        _clip_tr = _enc_clip(images_train_complet, "CLIP : encodage train")
        _clip_va = _enc_clip(images_val, "CLIP : encodage val")
        _clip_te = _enc_clip(images_test, "CLIP : encodage test")
        tp.sauver_cache("brique4_emb_clip", tr=_clip_tr, va=_clip_va, te=_clip_te)
    else:
        _clip_tr, _clip_va, _clip_te = _cache_clip["tr"], _cache_clip["va"], _cache_clip["te"]

    emb_clip_train_complet = _clip_tr
    emb_clip_train = _clip_tr[indices_gardes]
    emb_clip_val = _clip_va
    emb_clip_test = _clip_te

    _cache_dino = tp.charger_cache("brique4_emb_dino")
    if _cache_dino is None:
        _m_dino, _prep_dino = tp.charger_dino()
        _m_dino.eval()
        _appareil = tp.appareil()
        _m_dino.to(_appareil)

        def _enc_dino(imgs, titre):
            vecs = []
            with torch.no_grad():
                for i in mo.status.progress_bar(range(0, len(imgs), 32), title=titre):
                    lot = [_prep_dino(tp.Image.open(p).convert("RGB")) for p in imgs[i : i + 32]]
                    tenseur = torch.stack(lot).to(_appareil)
                    out = _m_dino(tenseur)
                    if hasattr(out, "last_hidden_state"):
                        out = out.last_hidden_state[:, 0]
                    elif isinstance(out, (tuple, list)):
                        out = out[0]
                    vecs.append(out.cpu().numpy())
            return np.concatenate(vecs, axis=0)

        _dino_tr = _enc_dino(images_train_complet, "DINOv2 : encodage train")
        _dino_va = _enc_dino(images_val, "DINOv2 : encodage val")
        _dino_te = _enc_dino(images_test, "DINOv2 : encodage test")
        tp.sauver_cache("brique4_emb_dino", tr=_dino_tr, va=_dino_va, te=_dino_te)
    else:
        _dino_tr, _dino_va, _dino_te = _cache_dino["tr"], _cache_dino["va"], _cache_dino["te"]

    emb_dino_train_complet = _dino_tr
    emb_dino_train = _dino_tr[indices_gardes]
    emb_dino_val = _dino_va
    emb_dino_test = _dino_te

    _val_class_full, _val_class_full_c = evaluer_recherche(emb_class_val, y_val, emb_class_train_complet, y_train_complet, noms, k=5)
    _val_clip_full, _val_clip_full_c = evaluer_recherche(emb_clip_val, y_val, emb_clip_train_complet, y_train_complet, noms, k=5)
    _val_dino_full, _val_dino_full_c = evaluer_recherche(emb_dino_val, y_val, emb_dino_train_complet, y_train_complet, noms, k=5)

    _val_class_rare, _val_class_rare_c = evaluer_recherche(emb_class_val, y_val, emb_class_train, y_train, noms, k=5)
    _val_clip_rare, _val_clip_rare_c = evaluer_recherche(emb_clip_val, y_val, emb_clip_train, y_train, noms, k=5)
    _val_dino_rare, _val_dino_rare_c = evaluer_recherche(emb_dino_val, y_val, emb_dino_train, y_train, noms, k=5)

    _tableau_val = [
        {
            "Méthode": "Baseline classique",
            "P@5 globale (180/classe)": _val_class_full,
            f"P@5 {CLASSE_RARE} (180)": _val_class_full_c[CLASSE_RARE],
            "P@5 globale (rare = 15)": _val_class_rare,
            f"P@5 {CLASSE_RARE} (15)": _val_class_rare_c[CLASSE_RARE],
        },
        {
            "Méthode": "CLIP (ViT-B/32)",
            "P@5 globale (180/classe)": _val_clip_full,
            f"P@5 {CLASSE_RARE} (180)": _val_clip_full_c[CLASSE_RARE],
            "P@5 globale (rare = 15)": _val_clip_rare,
            f"P@5 {CLASSE_RARE} (15)": _val_clip_rare_c[CLASSE_RARE],
        },
        {
            "Méthode": "DINOv2",
            "P@5 globale (180/classe)": _val_dino_full,
            f"P@5 {CLASSE_RARE} (180)": _val_dino_full_c[CLASSE_RARE],
            "P@5 globale (rare = 15)": _val_dino_rare,
            f"P@5 {CLASSE_RARE} (15)": _val_dino_rare_c[CLASSE_RARE],
        },
    ]

    modele_retenu = "DINOv2" if _val_dino_rare_c[CLASSE_RARE] >= _val_clip_rare_c[CLASSE_RARE] else "CLIP"

    mo.vstack([
        mo.md(f"### Comparaison sur la validation (modèle retenu : **{modele_retenu}**)"),
        tp.tableau(_tableau_val, mo),
    ])
    return (
        emb_clip_test,
        emb_clip_train,
        emb_dino_test,
        emb_dino_train,
        modele_retenu,
    )


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
def _(
    CLASSE_RARE,
    emb_class_test,
    emb_class_train,
    emb_clip_test,
    emb_clip_train,
    emb_dino_test,
    emb_dino_train,
    evaluer_recherche,
    idx_classe_rare,
    images_test,
    images_train,
    mo,
    modele_retenu,
    noms,
    np,
    rechercher,
    score_baseline,
    test_class_c,
    tp,
    y_test,
    y_train,
):
    # Une ligne par méthode, la même métrique pour toutes ; ajoutez ensuite vos figures et vos cas d'échec.
    _cache_res = tp.charger_cache("brique4_resultats_test")
    if _cache_res is None:
        _test_clip_glob, _test_clip_c = evaluer_recherche(emb_clip_test, y_test, emb_clip_train, y_train, noms, k=5)
        _test_dino_glob, _test_dino_c = evaluer_recherche(emb_dino_test, y_test, emb_dino_train, y_train, noms, k=5)
        score_modele = _test_clip_glob if modele_retenu == "CLIP" else _test_dino_glob
        score_rare_modele = _test_clip_c[CLASSE_RARE] if modele_retenu == "CLIP" else _test_dino_c[CLASSE_RARE]
        tp.sauver_cache(
            "brique4_resultats_test",
            score_modele=np.array(score_modele),
            score_rare_modele=np.array(score_rare_modele),
        )
    else:
        score_modele = float(_cache_res["score_modele"])
        score_rare_modele = float(_cache_res["score_rare_modele"])


    tableau_final = [
        {"méthode": "baseline classique", "score sur le test": score_baseline, f"P@5 {CLASSE_RARE}": test_class_c[CLASSE_RARE]},
        {"méthode": f"deep learning ({modele_retenu})", "score sur le test": score_modele, f"P@5 {CLASSE_RARE}": score_rare_modele},
    ]

    _idx_req = int(np.flatnonzero(np.asarray(y_test) == idx_classe_rare)[0])
    _top_clip = rechercher(emb_clip_test[_idx_req], emb_clip_train, k=5)
    _top_class = rechercher(emb_class_test[_idx_req], emb_class_train, k=5)

    _lire = lambda i: tp.lire_image_rgb(images_train[i], cote_max=200)
    _etiquette = lambda i: noms[y_train[i]]

    mo.vstack([
        tp.tableau(tableau_final, mo),
        tp.grille([tp.lire_image_rgb(images_test[_idx_req], cote_max=200)], [f"Requête : {CLASSE_RARE}"], n_col=1, taille=2.0),
        mo.md("**Top 5 — CLIP**"),
        tp.grille([_lire(i) for i in _top_clip], [_etiquette(i) for i in _top_clip], n_col=5, taille=1.8),
        mo.md("**Top 5 — Baseline classique**"),
        tp.grille([_lire(i) for i in _top_class], [_etiquette(i) for i in _top_class], n_col=5, taille=1.8),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Fiche de décision

    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | ... |
    | Baseline classique et son score | ... |
    | Modèle retenu et pourquoi | ... |
    | Métrique principale et score obtenu | ... |
    | Seuil ou réglage retenu, justification métier | ... |
    | Coût (annotation, entraînement, inférence, matériel) | ... |
    | Principaux cas d'échec | ... |
    | Le deep learning est-il justifié ici ? | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... |
    """)
    return


if __name__ == "__main__":
    app.run()
