"""One diagnostic candidate-generation and cross-play cycle, without training."""

from dataclasses import dataclass, replace
import hashlib

from .evaluator import CrossPlay, TeamSpec, evaluate_crossplay
from .search import CardVariant, LegalInventory, SearchConfig, SearchResult, generate_candidates


@dataclass(frozen=True)
class DiagnosticCycle:
    search_a: SearchResult
    search_b: SearchResult
    crossplay: CrossPlay


def run_diagnostic_cycle(model, card_embeddings, catalog, *, seed_team_a, seed_team_b,
                         inventory_a, inventory_b, search_config=None,
                         pack_ids=None, class_weights=None, identity_keys=None,
                         red_support_ids=(0, 0), blue_support_ids=(0, 0),
                         simulation_options=None, backend="c", library_path=None,
                         first_side=0, max_matches=4096):
    """Same policy for both sides; distinct reproducible search seeds.

    Each side optimizes against the other's initial lineup; all generated
    candidates then cross-play. Supports stay fixed. Inventory must be explicit
    for each side; screenshot inference never establishes legality here.
    Returns diagnostics only. CrossPlay.training_rows enforces the label gate.
    """
    config = SearchConfig() if search_config is None else search_config
    if type(first_side) is not int or first_side != 0:
        raise ValueError("Search currently assumes side A acts first; the model has no initiative input")
    if not all(isinstance(inv, LegalInventory) for inv in (inventory_a, inventory_b)):
        raise ValueError("Supply explicit LegalInventory objects for both sides")
    seed_team_a, seed_team_b = tuple(seed_team_a), tuple(seed_team_b)
    if any(len(team) != 4 or any(not isinstance(v, CardVariant) for v in team) for team in (seed_team_a, seed_team_b)):
        raise ValueError("Seed teams must contain exactly four CardVariant entries")
    if not isinstance(config, SearchConfig):
        raise ValueError("search_config must be SearchConfig")
    if type(max_matches) is not int or max_matches < 1 or config.candidate_count**2 > max_matches:
        raise ValueError("candidate_count squared must fit max_matches")
    # This is the executable subset pipeline: fail before costly search if even
    # one inventory variant or fixed support cannot be simulated completely.
    from ..simulator.catalog_rules import compile_battle, compile_fighter
    for inventory in (inventory_a, inventory_b):
        for variant in set(v for slot in inventory.slots for v in slot):
            compile_fighter(catalog, variant.card_id, variant.border_id, mutation=variant.mutation)
    compile_battle(catalog, (tuple(v.card_id for v in seed_team_a), tuple(v.card_id for v in seed_team_b)),
                   borders=(tuple(v.border_id for v in seed_team_a), tuple(v.border_id for v in seed_team_b)),
                   mutations=(tuple(v.mutation for v in seed_team_a), tuple(v.mutation for v in seed_team_b)),
                   red_supports=red_support_ids, blue_supports=blue_support_ids, first_side=first_side)
    shared = dict(red_support_ids=red_support_ids, blue_support_ids=blue_support_ids,
                  pack_ids=pack_ids, class_weights=class_weights, identity_keys=identity_keys,
                  intrinsic_weathers=tuple(catalog.weather(card.weather_id).name if card.weather_id != 1 else "None"
                                            for card in catalog.cards))
    seed_b = int.from_bytes(hashlib.blake2b(f"{config.seed}:side_b".encode(), digest_size=8).digest(), "little")
    result_a = generate_candidates(model, card_embeddings, seed_team=seed_team_a,
                 opponent_team=seed_team_b, inventory=inventory_a, target_side=0, config=config, **shared)
    result_b = generate_candidates(model, card_embeddings, seed_team=seed_team_b,
                 opponent_team=seed_team_a, inventory=inventory_b, target_side=1,
                 config=replace(config, seed=seed_b), **shared)

    def teams(result, side):
        return tuple(TeamSpec(tuple(v.card_id for v in row.team), tuple(v.border_id for v in row.team),
                              red_support_ids[side], blue_support_ids[side], tuple(v.mutation for v in row.team)) for row in result.candidates)

    crossplay = evaluate_crossplay(catalog, teams(result_a, 0), teams(result_b, 1),
                    options=simulation_options, backend=backend, library_path=library_path,
                    first_side=first_side, max_matches=max_matches)
    return DiagnosticCycle(result_a, result_b, crossplay)
