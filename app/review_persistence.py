"""Gradio adapters for the explicit, locally authorized persistence operations."""

from copy import deepcopy
import logging

import gradio as gr

import invoice_review
import invoice_review_edit as edit
from invoice_review_store import (
    StoreError, digest, document_identity, draft_data, snapshot_for,
)

LOGGER = logging.getLogger(__name__)


class PersistenceUI:
    def __init__(self, boundary, main):
        self.boundary, self.main = boundary, main

    def extract(self, filename, state):
        self.boundary.authorize()
        identity = document_identity(self.main.checked_path(filename))
        capture = {}
        for frame in self.main.run_extraction_ui_with_review(filename, state, capture=capture):
            if frame[0] is not None:
                if identity != document_identity(self.main.checked_path(filename)):
                    raise gr.Error("Le document a changé pendant l'extraction.")
                session = frame[11]
                snapshot = snapshot_for(
                    session["human_edit"], identity,
                    capture.get("provenance", {}),
                )
                session["persistence"] = {
                    "snapshot": snapshot, "snapshot_hash": digest(snapshot),
                    "draft_id": None, "revision": None, "saved_hash": None,
                    "warnings": [], "verified": True,
                }
            yield frame

    def status(self, session):
        self.boundary.authorize()
        metadata = (session or {}).get("persistence")
        if not metadata:
            return "Aucun brouillon enregistré. Les données sensibles ne sont sauvegardées que sur demande."
        state = session["human_edit"]
        dirty = metadata["saved_hash"] != digest(draft_data(state))
        identity = metadata["snapshot"]["document"]
        association = "document vérifié" if metadata["verified"] else "document à rattacher — inspection seule"
        warnings = "\n".join(str(value) for value in metadata["warnings"]) or "Aucune."
        return (
            f"Brouillon : {metadata['draft_id'] or 'nouveau'} · Révision : "
            f"{metadata['revision'] or 'non enregistrée'}\n\n"
            f"{'Modifications non enregistrées' if dirty else 'Enregistré'} · {association}\n\n"
            f"SHA-256 source : `{identity['sha256']}`\n\n"
            f"Confirmations à revoir / avertissements : {warnings}"
        )

    def refresh(self):
        self.boundary.authorize()
        rows = self.boundary.store.list_drafts()
        return gr.update(
            choices=[(f"{row['id']} · r{row['revision']} · {row['modified']}", row["id"])
                     for row in rows], value=None,
        )

    def save(self, session, consent):
        self.boundary.authorize()
        try:
            if consent is not True:
                raise StoreError("Confirmez le stockage local de données sensibles avant l'enregistrement.")
            if not isinstance(session, dict) or not session.get("persistence"):
                raise StoreError("Lancez une extraction ou ouvrez un brouillon.")
            metadata = session["persistence"]
            if not metadata["verified"]:
                raise StoreError("Rattachez d'abord le document source correspondant.")
            if digest(metadata["snapshot"]) != metadata["snapshot_hash"]:
                raise StoreError("Extraction snapshot mismatch.")
            saved = self.boundary.store.save(
                metadata["snapshot"], session["human_edit"],
                draft_id=metadata["draft_id"], revision=metadata["revision"],
            )
            updated = deepcopy(session)
            updated["persistence"].update(saved)
            message = "Enregistrement validé.\n\n" + self.status(updated)
            try:
                choices = self.refresh()
            except StoreError:
                choices = gr.update()
                message += "\n\nActualisation de liste impossible ; utilisez Actualiser."
                LOGGER.warning("Draft committed but list refresh failed")
            return updated, message, choices
        except (StoreError, edit.EditStateError, OSError) as exc:
            LOGGER.warning("Local draft save refused (%s)", type(exc).__name__)
            return session, str(exc), gr.update()

    def open(self, session, draft_id, replace):
        self.boundary.authorize()
        try:
            if not isinstance(draft_id, str):
                raise StoreError("Choisissez un brouillon enregistré.")
            if session and session.get("human_edit") and replace is not True:
                raise StoreError("Confirmez le remplacement du brouillon de session.")
            loaded = self.boundary.store.open(draft_id)
            snapshot = loaded["snapshot"]
            updated = {
                "pages": [], "limitations": ["Document source à rattacher."],
                "document": None, "human_edit": loaded["human_edit"],
                "persistence": {
                    **{key: loaded[key] for key in
                       ("draft_id", "revision", "saved_hash", "warnings")},
                    "snapshot": snapshot, "snapshot_hash": digest(snapshot),
                    "verified": False,
                },
            }
            review = snapshot["review"]
            return (
                updated, snapshot["strict"], [], {
                    "headers": list(invoice_review.COLUMN_HEADERS),
                    "data": invoice_review.review_rows(review, provenance=snapshot["provenance"]),
                },
                invoice_review.review_summary_markdown(
                    review, provenance=snapshot["provenance"],
                    limitations=("Document source à rattacher.",),
                ),
                review, *self.main._edit_outputs(updated), self.status(updated),
                None, None, "Brouillon enregistré ouvert — aucune nouvelle extraction.",
                "", "", "", {},
            )
        except (StoreError, edit.EditStateError, OSError) as exc:
            LOGGER.warning("Local draft open refused (%s)", type(exc).__name__)
            return (gr.skip(),) * 17 + (str(exc),) + (gr.skip(),) * 7

    def attach(self, session, filename):
        self.boundary.authorize()
        try:
            if not session or not session.get("persistence"):
                raise StoreError("Ouvrez d'abord un brouillon enregistré.")
            metadata = session["persistence"]
            # Opening another extraction is never an implicit merge operation.
            identity = document_identity(self.main.checked_path(filename))
            if identity != metadata["snapshot"]["document"]:
                raise StoreError("Source document mismatch: SHA-256 différent.")
            if metadata["draft_id"] is None:
                raise StoreError("Ce brouillon provient déjà de l'extraction courante.")
            loaded = self.boundary.store.open(metadata["draft_id"], document=identity)
            if loaded["revision"] != metadata["revision"]:
                raise StoreError("Une autre session a changé ce brouillon. Ouvrez-le à nouveau.")
            if digest(draft_data(session["human_edit"])) != metadata["saved_hash"]:
                raise StoreError("Rouvrez le brouillon : la session contient des modifications non enregistrées.")
            pages, limitations, name = self.main.review_pages(filename)
            updated = dict(session, pages=pages, limitations=list(limitations), document=name)
            updated["human_edit"] = loaded["human_edit"]
            updated["persistence"] = dict(
                metadata, verified=True, warnings=loaded["warnings"],
                saved_hash=loaded["saved_hash"],
            )
            return updated, pages, *self.main._edit_outputs(updated), self.status(updated)
        except (StoreError, edit.EditStateError, OSError) as exc:
            LOGGER.warning("Local draft document attachment refused (%s)", type(exc).__name__)
            return (gr.skip(),) * 13 + (str(exc),)

    def delete(self, session, confirmed):
        self.boundary.authorize()
        try:
            if not session or not session.get("persistence"):
                raise StoreError("Ouvrez le brouillon à supprimer.")
            metadata = session["persistence"]
            self.boundary.store.delete(
                metadata["draft_id"], metadata["revision"], confirmed=confirmed,
            )
            try:
                choices = self.refresh()
            except StoreError:
                choices = gr.update()
                LOGGER.warning("Draft deleted but list refresh failed")
            # Retain the old ID/revision: a stale save must fail, not create a new record.
            return (
                session, "Copie enregistrée supprimée. La session n'est pas effacée. "
                "Toute sauvegarde de cette ancienne révision sera refusée.",
                choices,
            )
        except (StoreError, OSError) as exc:
            LOGGER.warning("Local draft deletion refused (%s)", type(exc).__name__)
            return session, str(exc), gr.update()
