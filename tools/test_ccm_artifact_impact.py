"""Synthetic-only tests for the read-only cross-shard impact inventory."""
import importlib.util
import json
import re
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("impact", Path(__file__).with_name("ccm_artifact_impact.py"))
impact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(impact)


def artifact(root, digit, kind, version, dependencies=(), active=True, parameters=None, algorithm=None):
    if parameters is None:
        parameters = dict(impact.CURRENT_PARAMETERS.get(kind, {})) if isinstance(kind,str) else {}
        if kind == 'ccm_sector_spectrum':
            parameters['eigenvector_iteration_limit'] = 256
        if isinstance(kind,str) and kind in impact.CURRENT_REQUEST_PARAMETERS:
            parameters['request'] = dict(impact.CURRENT_REQUEST_PARAMETERS[kind])
        if isinstance(kind,str) and kind in impact.ROOT_KINDS:
            parameters.update(solver='newton', fixed_guard_correction_guards=[64,128,256,512,1024,2048,4096])
    if isinstance(kind,str) and kind == 'ccm_indexed_transform_analysis' and not parameters:
        parameters['request']={'formula':'integral_-L/2^L/2 f(x)*exp(i*t*x) dx; analytic_sinc_and_derivative','transform_arithmetic':'exact_cutoff_stored_points_directed_sinc_v1'}
    if algorithm is None:
        if isinstance(kind,str) and kind in impact.CURRENT_ALGORITHMS:
            algorithm = sorted(impact.CURRENT_ALGORITHMS[kind])[0]
        elif isinstance(kind,str) and kind in impact.ROOT_KINDS:
            algorithm = 'newton+confirmed_retained_correction_v2'
    digest = digit * 64
    path = root / "manifests" / digest[:2] / (digest + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artifact_family": "ccm-evidence",
        "semantic_key": {"artifact_kind": kind, "mathematical_semantics_version": version,
                         "resolved_mathematical_parameters": parameters, "algorithm_semantics": algorithm},
        "canonical_payload": {"dependencies": [{"manifest_digest": d * 64} for d in dependencies]},
    }))
    index = root / "indexes" / "ccm-evidence" / (digit + ".json")
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"entries": [{"manifest_digest": digest, "disposition": "active" if active else "retired"}]}))
    return digest


def rust_end_brace(text, opening):
    """Bound a reviewed Rust block; quotes can contain braces."""
    depth, quoted, escaped = 1, False, False
    for i in range(opening+1,len(text)):
        c=text[i]
        if quoted:
            if escaped: escaped=False
            elif c == '\\': escaped=True
            elif c == '"': quoted=False
        elif c == '"': quoted=True
        elif c == '{': depth+=1
        elif c == '}':
            depth-=1
            if not depth: return i
    raise AssertionError('unclosed source block')

def rust_body(source, signature):
    start=source.index('{',source.index(signature))
    return source[start+1:rust_end_brace(source,start)]


class ImpactTests(unittest.TestCase):
    def test_response_upgrade_preserves_sources_and_counts_receipt_descendants(self):
        artifact(self.root, "a", 'ccm_weil_eigenpair', 'ccm-smallest-weil-eigenpair-stored-resolution-v4')
        artifact(self.root, "b", "ccm_u_flow_response_analysis", "ccm-u-flow-response-v0.14.1-v2", ["a"])
        artifact(self.root, "c", "ccm_prime_power_response_analysis", "ccm-prime-power-response-v0.14.1-v2", ["a"])
        artifact(self.root, "d", 'research_capture_receipt', 'managed-research-record-v1', ["b", "c"])
        artifact(self.root, "e", "ccm_u_flow_response_analysis", impact.CURRENT_IDENTITIES["ccm_u_flow_response_analysis"], ["a"])
        report = impact.inventory([self.root])
        self.assertEqual(report["active_recompute_count"], 3)
        self.assertEqual(report["active_affected_count"], 0)  # certificate-only defect rule
        self.assertNotIn("ccm_weil_eigenpair", report["active_recompute_by_kind"])

    def test_stable_flow_epoch_requires_new_flow_and_receipt_identities(self):
        artifact(self.root, "a", 'ccm_weil_eigenpair', 'ccm-smallest-weil-eigenpair-stored-resolution-v4')
        artifact(self.root, "b", "ccm_u_flow_response_analysis", "ccm-u-flow-response-v0.15.0-v3", ["a"])
        artifact(self.root, "c", 'research_capture_receipt', 'managed-research-record-v1', ["b"])
        artifact(self.root, "d", "ccm_u_flow_response_analysis", impact.CURRENT_IDENTITIES["ccm_u_flow_response_analysis"], ["a"])
        report = impact.inventory([self.root])
        self.assertEqual(report["active_recompute_count"], 2)
        self.assertEqual(report["active_affected_count"], 0)
        self.assertEqual(report["active_recompute_by_kind"], {"ccm_u_flow_response_analysis": 1, "research_capture_receipt": 1})

    def test_source_isolation_epoch_requires_new_response_and_receipt_identities(self):
        artifact(self.root, "a", 'ccm_weil_eigenpair', 'ccm-smallest-weil-eigenpair-stored-resolution-v4')
        artifact(self.root, "b", "ccm_u_flow_response_analysis", "ccm-u-flow-response-v0.15.1-v4", ["a"])
        artifact(self.root, "c", "ccm_prime_power_response_analysis", "ccm-prime-power-response-v0.15.0-v3", ["a"])
        artifact(self.root, "d", 'research_capture_receipt', 'managed-research-record-v1', ["b", "c"])
        artifact(self.root, "e", "ccm_u_flow_response_analysis", impact.CURRENT_IDENTITIES["ccm_u_flow_response_analysis"], ["a"])
        artifact(self.root, "f", "ccm_prime_power_response_analysis", impact.CURRENT_IDENTITIES["ccm_prime_power_response_analysis"], ["a"])
        report = impact.inventory([self.root])
        self.assertEqual(report["active_recompute_count"], 3)
        self.assertEqual(report["active_affected_count"], 0)
        self.assertNotIn("ccm_weil_eigenpair", report["active_recompute_by_kind"])

    def test_upgrade_rules_count_descendants_without_calling_them_corrupt(self):
        artifact(self.root, "a", "ccm_sector_tridiagonal", "ccm-parity-tridiagonal-v0.13.0-v3")
        artifact(self.root, "b", "ccm_sector_eigenvalues", "unchanged-v1", ["a"])
        artifact(self.root, "c", "ccm_tau_matrix", impact.CURRENT_IDENTITIES["ccm_tau_matrix"])
        artifact(self.root, "d", "ccm_sector_spectrum", impact.CURRENT_IDENTITIES["ccm_sector_spectrum"])
        report = impact.inventory([self.root])
        self.assertEqual(report["active_recompute_count"], 2)
        self.assertEqual(report["active_affected_count"], 0)
        self.assertEqual(report["active_recompute_by_kind"], {"ccm_sector_tridiagonal": 1, "ccm_sector_eigenvalues": 1})

    def test_registry_exposes_absent_rollover_shards(self):
        registry = self.root / "registry"
        (registry / "shards").mkdir(parents=True)
        (registry / "shards" / "0002.json").write_text(json.dumps({"repository": "synthetic/matrices-0002"}))
        report = impact.inventory([self.root], registries=[registry])
        self.assertEqual(report["missing_registry_repositories"], ["synthetic/matrices-0002"])
        self.assertFalse(report["coverage_complete_for_supplied_shards"])

    def test_invalid_registry_cannot_appear_complete(self):
        registry = self.root / "registry"
        (registry / "shards").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "no shard descriptors"):
            impact.inventory([self.root], registries=[registry])
        (registry / "shards" / "bad.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "no repository identity"):
            impact.inventory([self.root], registries=[registry])

    def test_exhaustive_current_prefix_and_reduction_epochs(self):
        artifact(self.root,'a','ccm_prefix_analysis','ccm-retained-even-prefix-moments-checked-exports-v8')
        artifact(self.root,'b','ccm_prefix_analysis','ccm-retained-even-prefix-moments-checked-exports-v10')
        artifact(self.root,'c','ccm_retained_reduction_check','ccm-retained-reduction-v0.15.1-v3')
        artifact(self.root,'d','ccm_prefix_analysis','ccm-retained-even-prefix-moments-checked-exports-v6')
        report=impact.inventory([self.root])
        flagged={row['manifest_digest'][0] for row in report['artifacts'] if row['upgrade_reuse']=='new_semantic_identity_required'}
        self.assertEqual(flagged,{'d'})

    def test_current_root_epochs_use_fresh_independent_producer_records(self):
        records = json.loads((Path(__file__).with_name('fixtures') / 'current-root-semantic-keys.json').read_text(encoding='utf-8'))
        for route, original in records.items():
            with self.subTest(route=route):
                artifact(self.root, 'a', original['artifact_kind'], original['mathematical_semantics_version'],
                         parameters=original['resolved_mathematical_parameters'], algorithm=original['algorithm_semantics'])
                report = impact.inventory([self.root])
                self.assertEqual(report['active_recompute_count'], 0)
                for marker in ['point_refinement_semantics', 'generic_discovery_semantics']:
                    old = dict(original['resolved_mathematical_parameters'])
                    old.pop(marker)
                    artifact(self.root, 'b', original['artifact_kind'], original['mathematical_semantics_version'],
                             parameters=old, algorithm=original['algorithm_semantics'])
                    self.assertEqual(impact.inventory([self.root])['active_recompute_count'], 1)
                artifact(self.root, 'b', original['artifact_kind'], original['mathematical_semantics_version'],
                         parameters=original['resolved_mathematical_parameters'], algorithm=original['algorithm_semantics'])

    def test_adaptive_policy_is_read_from_independent_recorded_producer_key(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/adaptive-root-semantic-key.json').read_text(encoding='utf-8'))['semantic_key']
        # The recorded key comes from the HP producer, independently of this tool.
        # Later numerical identities may flag other fields; adaptive requests
        # must never be tested as fixed-guard requests.
        for kind, identity in [('ccm_root_discovery_window','ccm-root-discovery-v0.15.2-v2'),
                               ('ccm_root_refinement','ccm-root-range-v0.15.2-v12')]:
            for solver in ['halley','newton']:
                key=json.loads(json.dumps(fixture))
                key['artifact_kind']=kind
                key['mathematical_semantics_version']=identity
                key['resolved_mathematical_parameters']['solver']=solver
                key['algorithm_semantics']=solver+'+adaptive_root_precision_v1'
                mismatches=impact.additional_identity_mismatches(key)
                self.assertNotIn('root_algorithm_semantics',mismatches)
                self.assertNotIn('fixed_guard_correction_guards',mismatches)
                key['algorithm_semantics']=solver+'+confirmed_retained_correction_v2'
                self.assertIn('root_algorithm_semantics',impact.additional_identity_mismatches(key))

    def test_exhaustive_root_response_epochs_are_current(self):
        artifact(self.root,'a','ccm_prime_power_response_analysis','ccm-prime-power-response-exact-event-edge-v3')
        artifact(self.root,'b','ccm_u_flow_response_analysis','ccm-u-flow-response-v0.15.2-v3')
        artifact(self.root,'c','ccm_prime_power_response_analysis','ccm-prime-power-response-v0.15.1-v11')
        artifact(self.root,'d','ccm_u_flow_response_analysis','ccm-u-flow-response-v0.15.1-v12')
        artifact(self.root,'e','ccm_u_flow_response_analysis','ccm-u-flow-response-exact-quadrants-v4')
        report=impact.inventory([self.root])
        flagged={row['manifest_digest'][0] for row in report['artifacts'] if row['upgrade_reuse']=='new_semantic_identity_required'}
        self.assertEqual(flagged,{'b','c','d'})

    def test_all_current_identity_rules_are_bound_to_producer_sources(self):
        root = Path(__file__).resolve().parents[1]
        sources = {
            'gauss_legendre_rule': '../../xc-numerics/src/quadrature.rs',
            **{kind: 'ccm/retained_evidence.rs' for kind in ['ccm_arithmetic_energy_analysis', 'ccm_band_reconstruction', 'ccm_capture_preflight', 'ccm_compactness_analysis', 'ccm_complex_transform_analysis', 'ccm_configuration_comparison', 'ccm_consistency_analysis', 'ccm_directional_response_analysis', 'ccm_energy_allowance_analysis', 'ccm_external_research_source', 'ccm_finite_section_transfer', 'ccm_indexed_transform_analysis', 'ccm_observable_budget_analysis', 'ccm_operator_cluster_analysis', 'ccm_operator_energy_analysis', 'ccm_reference_projection_analysis', 'ccm_reference_source', 'ccm_resolution_budget_analysis', 'ccm_root_band_analysis', 'ccm_root_transport_analysis', 'ccm_signed_transform_analysis', 'ccm_spectral_cluster_analysis', 'ccm_stabilization_analysis', 'ccm_tail_operator_analysis', 'ccm_transform_enclosure', 'ccm_weighted_reference_projection', 'ccm_weighted_tail_analysis', 'research_observation_packet', 'research_reference_dataset']},
            'ccm_state_geometry_analysis': 'ccm/state_geometry.rs',
            'ccm_sector_gap_certificate': 'ccm/sector_gap_certificate.rs',
            'research_capture_receipt': '../../xc-cache/src/research_artifacts.rs',
            'research_hypothesis_evaluation': '../../xc-cache/src/research_artifacts.rs',
            'ccm_discretization_distance': 'distance.rs',
            'ccm_prefix_analysis': 'ccm/prefix.rs',
            'ccm_retained_reduction_check': 'ccm/prefix.rs',
            'prolate_eigenvalue_spectrum': 'prolate.rs',
            **{kind: 'distance.rs' for kind in [
                'ccm_eigenfunction_profile', 'ccm_target_distance',
                'ccm_distance_resolution_evidence', 'ccm_target_residual_analysis',
                'ccm_deviation_decomposition',
            ]},
        }
        bound_children = {'ccm_distance_resolution_evidence', 'ccm_target_residual_analysis', 'ccm_deviation_decomposition'}
        for kind, current in impact.CURRENT_IDENTITIES.items():
            text = (root/'crates/xc-spectral/src'/sources.get(kind, 'ccm/hp.rs')).read_text(encoding='utf-8')
            for version in {current} | impact.ALTERNATE_IDENTITIES.get(kind, set()):
                base = version.removesuffix('-retained-parents-v3') if kind in bound_children else version
                self.assertIn('"'+base+'"', text, kind)
                if kind in bound_children:
                    self.assertIn('"{}-retained-parents-v3"', text, kind)

    def test_current_and_superseded_versions_from_independent_producer_census(self):
        # Producer-derived fixtures, including the binding suffix, not expected
        # values copied from the impact table being exercised.
        cases = [
            ('ccm_sector_eigenvalues', 'ccm-parity-sector-eigenvalues-isolating-v4', 'ccm-parity-sector-eigenvalues-isolating-v3'),
            ('ccm_sector_eigenvalues', 'ccm-even-response-lowest-isolation-with-neighbor-cluster-v1', 'ccm-parity-sector-eigenvalues-v0.15.0-v1'),
            ('ccm_u_flow_response_analysis', 'ccm-u-flow-response-exact-quadrants-v4', 'ccm-u-flow-response-v0.15.2-v3'),
            ('ccm_sector_spectrum', 'ccm-parity-sector-spectrum-source-enclosures-v3', 'ccm-parity-sector-spectrum-v0.15.1-v2'),
            ('ccm_eigenfunction_profile', 'ccm-eigenfunction-profile-checked-grid-and-coefficients-v0.15.1-v4', 'ccm-eigenfunction-profile-lossless-v0.15.0-v1'),
            ('ccm_target_distance', 'ccm-runtime-target-distance-exact-weight-checked-coefficients-v0.15.1-v4', 'ccm-runtime-target-distance-lossless-v0.15.0-v1'),
            ('prolate_eigenvalue_spectrum', 'prolate-fd-working-precision-spectrum-v0.15.1-v2', 'prolate-fd-exact-source-spectrum-v0.15.0-v1'),
            ('ccm_distance_resolution_evidence', 'ccm-runtime-target-resolution-evidence-roundtrip-tail-v7-retained-parents-v3', 'ccm-runtime-target-resolution-evidence-retained-verdict-v6'),
            ('ccm_target_residual_analysis', 'ccm-runtime-target-residual-analysis-roundtrip-exact-weight-v6-retained-parents-v3', 'ccm_target_residual_analysis-retained-parents-v0.15.0-v1'),
            ('ccm_deviation_decomposition', 'ccm-runtime-target-decomposition-roundtrip-v7-retained-parents-v3', 'ccm_deviation_decomposition-retained-parents-v0.15.0-v1'),
        ]
        for kind, current, previous in cases:
            artifact(self.root, 'a', kind, current)
            artifact(self.root, 'b', kind, previous)
            artifact(self.root, 'c', 'research_capture_receipt', 'managed-research-record-v1', ['b'])
            rows = {r['manifest_digest'][0]: r['upgrade_reuse'] for r in impact.inventory([self.root])['artifacts']}
            self.assertEqual(rows['a'], 'not_flagged_by_upgrade_rules', kind)
            self.assertEqual(rows['b'], 'new_semantic_identity_required', kind)
            self.assertEqual(rows['c'], 'new_parent_identity_required', kind)

    def test_route_specific_repairs_preserve_other_current_routes(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/'crates/xc-spectral/src/ccm/hp.rs').read_text(encoding='utf-8')
        for (kind, previous), current in impact.SUPERSEDED_IDENTITIES.items():
            self.assertIn('"'+current+'"', source)
            parameters = impact.CURRENT_PARAMETERS.get(kind, {})
            artifact(self.root, 'a', kind, previous, parameters=parameters)
            artifact(self.root, 'b', kind, current, parameters=parameters)
            artifact(self.root, 'c', 'research_capture_receipt', 'managed-research-record-v1', ['a'])
            artifact(self.root, 'd', kind, 'different-current-route', parameters=parameters)
            rows = {r['manifest_digest'][0]: r['upgrade_reuse'] for r in impact.inventory([self.root])['artifacts']}
            self.assertEqual(rows['a'], 'new_semantic_identity_required')
            self.assertEqual(rows['b'], 'not_flagged_by_upgrade_rules')
            self.assertEqual(rows['c'], 'new_parent_identity_required')
            self.assertEqual(rows['d'], 'new_semantic_identity_required')

    def test_exhaustive_ground_validation_parameter_changes_identity(self):
        digest=artifact(self.root,'a','ccm_weil_eigenpair', 'ccm-smallest-weil-eigenpair-stored-resolution-v4',parameters={})
        artifact(self.root,'b','research_capture_receipt', 'managed-research-record-v1',['a'])
        report=impact.inventory([self.root]);self.assertEqual(report['active_recompute_count'],2)
        path=self.root/'manifests'/digest[:2]/(digest+'.json');data=json.loads(path.read_text())
        data['semantic_key']['resolved_mathematical_parameters']={'inverse_iteration_semantics':'dense-inverse-iteration-mixed-parity-residual-checked-wrapper-v3','source_identity_arithmetic':'validated_tau_dependency_and_exact_log_source_v2','ground_index_validation':'directed_stored_source_ground_index_and_rounding_scale_gap_v2','tau_residual_arithmetic':'directed_scaled_stored_infinity_norm_v2','parity_basis_arithmetic':'directed_stored_point_orthonormal_parity_v2','state_normalization_arithmetic':'directed_scaled_exact_sum_sqrt_l_state_v2','l2_normalization_arithmetic':'power_two_scaled_max_precision_l2_v2'}
        path.write_text(json.dumps(data))
        self.assertEqual(impact.inventory([self.root])['active_recompute_count'],0)

    def test_exhaustive_malformed_manifest_kind_is_reported(self):
        artifact(self.root,'a',[], 'v1')
        report=impact.inventory([self.root]);self.assertTrue(report['metadata_errors'])
        self.assertFalse(report['coverage_complete_for_supplied_shards'])

    def test_exhaustive_index_failure_does_not_retain_partial_active_entries(self):
        digest=artifact(self.root,'a','ccm_tau_matrix','v1')
        path=self.root/'indexes'/'ccm-evidence'/'a.json'
        path.write_text(json.dumps({'entries':[{'manifest_digest':digest,'disposition':'active'},None]}))
        report=impact.inventory([self.root]);self.assertTrue(report['metadata_errors'])
        self.assertEqual(report['active_manifests'],0)

    def test_parity_projection_versions_require_recompute_with_descendants(self):
        artifact(self.root,'a','ccm_even_sector_matrix','ccm-even-sector-v0.13.0-v1')
        artifact(self.root,'b','ccm_odd_sector_matrix','ccm-odd-sector-v0.13.0-v1')
        artifact(self.root,'c','research_capture_receipt', 'managed-research-record-v1',['a','b'])
        for label,kind in [('d','ccm_even_sector_matrix'),('e','ccm_odd_sector_matrix')]:
            artifact(self.root,label,kind,impact.CURRENT_IDENTITIES[kind])
        report=impact.inventory([self.root])
        flagged={row['manifest_digest'][0] for row in report['artifacts'] if row['upgrade_reuse']=='new_semantic_identity_required'}
        self.assertEqual(flagged,{'a','b'})
        descendants={row['manifest_digest'][0] for row in report['artifacts'] if row['upgrade_reuse']=='new_parent_identity_required'}
        self.assertEqual(descendants,{'c'})
        self.assertEqual(report['active_recompute_count'],3)

    def test_tau_quadrature_identity_upgrade_propagates_to_eigenstates(self):
        artifact(self.root,'a','ccm_tau_matrix','ccm-weil-form-v0.13.0-v2')
        artifact(self.root,'b','ccm_weil_eigenpair', 'ccm-smallest-weil-eigenpair-stored-resolution-v4',['a'])
        artifact(self.root,'c','ccm_tau_matrix',impact.CURRENT_IDENTITIES['ccm_tau_matrix'])
        report=impact.inventory([self.root])
        rows={row['manifest_digest'][0]:row['upgrade_reuse'] for row in report['artifacts']}
        self.assertEqual(rows['a'],'new_semantic_identity_required')
        self.assertEqual(rows['b'],'new_parent_identity_required')
        self.assertEqual(rows['c'],'not_flagged_by_upgrade_rules')
        self.assertEqual(report['active_recompute_count'],2)

    def test_each_eigenstate_arithmetic_parameter_is_required(self):
        kind='ccm_weil_eigenpair'
        for key in impact.CURRENT_PARAMETERS[kind]:
            params=dict(impact.CURRENT_PARAMETERS[kind]);del params[key]
            artifact(self.root,'a',kind,impact.CURRENT_IDENTITIES[kind],parameters=params)
            self.assertEqual(impact.inventory([self.root])['active_recompute_count'],1,key)

    def test_complete_root_boundary_epoch_recomputes_old_window_and_descendants(self):
        artifact(self.root,'a','ccm_root_discovery_window','ccm-root-range-v0.15.1-v13')
        artifact(self.root,'b','research_capture_receipt', 'managed-research-record-v1',['a'])
        artifact(self.root,'c','ccm_root_discovery_window','ccm-root-range-v0.15.2-v15')
        for label,epoch in [('d','ccm-root-range-v0.15.2-v11'),('e','ccm-root-range-v0.15.2-v11-advanced'),('f','ccm-root-range-v0.15.2-v12')]:
            artifact(self.root,label,'ccm_root_discovery_window',epoch)
        report=impact.inventory([self.root]);rows={r['manifest_digest'][0]:r['upgrade_reuse'] for r in report['artifacts']}
        self.assertEqual(rows['a'],'new_semantic_identity_required')
        self.assertEqual(rows['b'],'new_parent_identity_required')
        self.assertEqual(rows['c'],'not_flagged_by_upgrade_rules')
        for label in ['d','e','f']:self.assertEqual(rows[label],'new_semantic_identity_required')
        self.assertEqual(report['active_recompute_count'],5)
        self.assertEqual(report['active_affected_count'],0)

    def test_upgrade_rules_match_current_source_constants(self):
        root=Path(__file__).resolve().parents[1]
        hp=(root/'crates/xc-spectral/src/ccm/hp.rs').read_text(encoding='utf-8')
        prefix=(root/'crates/xc-spectral/src/ccm/prefix.rs').read_text(encoding='utf-8')
        for kind in ['ccm_tau_matrix','ccm_even_sector_matrix','ccm_odd_sector_matrix','ccm_prime_power_response_analysis','ccm_u_flow_response_analysis']:
            self.assertIn('"'+impact.CURRENT_IDENTITIES[kind]+'"',hp)
        for kind in ['ccm_prefix_analysis','ccm_retained_reduction_check']:
            self.assertIn('"'+impact.CURRENT_IDENTITIES[kind]+'"',prefix)
        for value in impact.ALTERNATE_IDENTITIES['ccm_prefix_analysis']:
            self.assertIn('"'+value+'"',prefix)
        sources={'source_identity_arithmetic':'xc-spectral/src/ccm/hp.rs','parity_basis_arithmetic':'xc-spectral/src/ccm/hp/parity_math.rs','state_normalization_arithmetic':'xc-spectral/src/ccm/hp/state_normalization_math.rs','ground_index_validation':'xc-spectral/src/ccm/hp/ground_index.rs','tau_residual_arithmetic':'xc-spectral/src/ccm/hp/standalone_cache.rs','l2_normalization_arithmetic':'xc-numerics/src/linalg.rs'}
        for key,file in sources.items():
            self.assertIn('"'+impact.CURRENT_PARAMETERS['ccm_weil_eigenpair'][key]+'"',(root/'crates'/file).read_text(encoding='utf-8'))

    def test_fresh_audit_repair_epochs_propagate_without_claiming_corruption(self):
        artifact(self.root, 'a', 'ccm_root_discovery_window', 'ccm-root-range-v0.15.2-v14')
        artifact(self.root, 'b', 'ccm_root_discovery_window', 'ccm-root-range-v0.15.2-v15')
        artifact(self.root, 'c', 'ccm_distance_resolution_evidence', 'ccm-runtime-target-resolution-evidence-outward-v0.15.1-v5')
        artifact(self.root, 'd', 'ccm_distance_resolution_evidence', impact.CURRENT_IDENTITIES['ccm_distance_resolution_evidence'])
        artifact(self.root, 'e', 'research_capture_receipt', 'managed-research-record-v1', ['a', 'c'])
        artifact(self.root, 'f', 'ccm_arithmetic_energy_analysis', 'ccm-retained-research-observations-v22', parameters={'request': {}})
        artifact(self.root, '1', 'ccm_arithmetic_energy_analysis', 'ccm-retained-research-observations-v22', parameters={'request': impact.CURRENT_REQUEST_PARAMETERS['ccm_arithmetic_energy_analysis']})
        report = impact.inventory([self.root])
        rows = {r['manifest_digest'][0]: r['upgrade_reuse'] for r in report['artifacts']}
        self.assertEqual({k for k,v in rows.items() if v == 'new_semantic_identity_required'}, {'a','c','f'})
        self.assertEqual(rows['e'], 'new_parent_identity_required')
        self.assertEqual(report['active_affected_count'], 0)
        self.assertEqual(report['active_recompute_count'], 4)

    def test_unknown_kinds_cannot_claim_complete_upgrade_coverage(self):
        artifact(self.root, 'a', 'new_unreviewed_producer_kind', 'v1')
        report = impact.inventory([self.root])
        self.assertTrue(report['metadata_coverage_complete_for_supplied_shards'])
        self.assertFalse(report['upgrade_rule_coverage_complete_for_supplied_kinds'])
        self.assertFalse(report['coverage_complete_for_supplied_shards'])
        self.assertEqual(report['active_uncovered_kind_count'], 1)
        self.assertEqual(report['artifacts'][0]['upgrade_reuse'], 'unrecognized_artifact_kind_requires_review')
        self.assertEqual(report['active_recompute_count'], 0)  # Unknown is not proof of obsolescence.

    def test_omitted_kinds_and_corrected_but_superseded_certificate_recompute(self):
        cases = [
            ('ccm_root_conditioning_analysis', 'ccm-root-conditioning-v0.15.2-v1'),
            ('ccm_discretization_distance', 'ccm-discretization-distance-roundtrip-pinned-grid-v6'),
            ('ccm_sector_gap_certificate', 'ccm-cutoff-free-sector-gap-certificate-mpfr-selected-v5'),
            ('ccm_compactness_analysis', 'ccm-retained-research-observations-v22'),
        ]
        for kind, current in cases:
            with self.subTest(kind=kind):
                previous = 'ccm-cutoff-free-sector-gap-certificate-v0.15.0-v3' if kind == 'ccm_sector_gap_certificate' else 'manufactured-older-version'
                artifact(self.root, 'a', kind, current)
                artifact(self.root, 'b', kind, previous)
                artifact(self.root, 'c', 'research_capture_receipt', 'managed-research-record-v1', ['b'])
                report = impact.inventory([self.root])
                rows = {r['manifest_digest'][0]: r for r in report['artifacts']}
                self.assertEqual(rows['a']['upgrade_reuse'], 'not_flagged_by_upgrade_rules')
                self.assertEqual(rows['b']['upgrade_reuse'], 'new_semantic_identity_required')
                self.assertEqual(rows['c']['upgrade_reuse'], 'new_parent_identity_required')
                self.assertEqual(report['active_recompute_count'], 2)
                self.assertEqual(report['active_affected_count'], 0)

    def test_current_version_does_not_hide_changed_algorithm_or_grid_arithmetic(self):
        cases = [
            ('ccm_sector_eigenvalues', {}, 'old_qr'),
            ('ccm_root_refinement', {'solver':'newton'}, 'newton'),
            ('ccm_root_conditioning_analysis', {}, 'old_conditioning'),
            ('ccm_target_distance', {}, None),
            ('ccm_weil_eigenpair', {k:v for k,v in impact.CURRENT_PARAMETERS['ccm_weil_eigenpair'].items() if k != 'inverse_iteration_semantics'}, None),
        ]
        for kind, parameters, algorithm in cases:
            with self.subTest(kind=kind):
                artifact(self.root, 'a', kind, impact.CURRENT_IDENTITIES[kind], parameters=parameters, algorithm=algorithm)
                self.assertEqual(impact.inventory([self.root])['active_recompute_count'], 1)

    def test_each_current_route_and_each_known_arithmetic_marker(self):
        for kind, primary in impact.CURRENT_IDENTITIES.items():
            for current in {primary} | impact.ALTERNATE_IDENTITIES.get(kind,set()):
                with self.subTest(kind=kind, version=current):
                    artifact(self.root, 'a', kind, current)
                    self.assertEqual(impact.inventory([self.root])['active_recompute_count'], 0)
            artifact(self.root, 'a', kind, 'manufactured-noncurrent-label')
            self.assertEqual(impact.inventory([self.root])['active_recompute_count'], 1, kind)
        for kind, required in impact.CURRENT_REQUEST_PARAMETERS.items():
            for key in required:
                request = dict(required)
                del request[key]
                artifact(self.root, 'a', kind, impact.CURRENT_IDENTITIES[kind], parameters={'request':request})
                self.assertEqual(impact.inventory([self.root])['active_recompute_count'], 1, (kind,key))

    def test_current_kind_map_has_independent_production_source_census(self):
        # This census reads the producing envelopes/callers, not arbitrary
        # historical strings elsewhere in each file. New literal producers must
        # be accounted for even when the impact table itself is unchanged.
        root = Path(__file__).resolve().parents[1]
        base = root/'crates/xc-spectral/src'
        for file in ['ccm/hp.rs','distance.rs','ccm/sector_gap_certificate.rs','../../xc-numerics/src/quadrature.rs']:
            source = (base/file).read_text(encoding='utf-8')
            for match in re.finditer(r'artifact_kind:\s*"([a-z_]+)"\.to_owned\(\),\s*mathematical_semantics_version:\s*(.*?)\s*,\s*resolved_mathematical_parameters(?::|,)', source, re.S):
                kind, expression = match.groups()
                if kind.startswith('fixture'):
                    continue
                self.assertIn(kind, impact.CURRENT_IDENTITIES, (file, kind))
                emitted = set(re.findall(r'"((?:ccm|prolate|gauss-legendre)-[^"\n]+)"', expression))
                if not emitted:
                    constant = re.fullmatch(r'([A-Z_]+)\.to_owned\(\)', expression)
                    self.assertIsNotNone(constant, expression)
                    value = re.search(r'const\s+'+constant.group(1)+r':\s*&str\s*=\s*"([^"]+)"',source)
                    self.assertIsNotNone(value, constant.group(1))
                    emitted = {value.group(1)}
                if kind in {'ccm_distance_resolution_evidence','ccm_target_residual_analysis','ccm_deviation_decomposition'}:
                    self.assertIn('format!("{}-retained-parents-v3", key.mathematical_semantics_version)', source)
                    emitted = {value+'-retained-parents-v3' for value in emitted}
                expected = {impact.CURRENT_IDENTITIES[kind]} | impact.ALTERNATE_IDENTITIES.get(kind,set())
                self.assertEqual(emitted, expected, (file,kind))
        retained = (base/'ccm/retained_evidence.rs').read_text(encoding='utf-8')
        extended = (base/'ccm/extended_research.rs').read_text(encoding='utf-8')
        current = re.search(r'pub const SEMANTICS:\s*&str\s*=\s*"([^"]+)"', retained).group(1)
        kinds = set(re.findall(r'\bmanaged\(\s*"([a-z_]+)"',retained))
        kinds.update(re.findall(r'"[a-z_]+"\s*=>\s*"(ccm_[a-z_]+)"', rust_body(extended,'pub fn artifact_kind')))
        kinds.add(re.search(r'pub const INPUT_KIND:\s*&str\s*=\s*"([^"]+)"',extended).group(1))
        for kind in kinds:
            self.assertEqual(impact.CURRENT_IDENTITIES.get(kind), current, kind)
        self.assertIn('mathematical_semantics_version: SEMANTICS.into()',rust_body(retained,'pub(super) fn managed'))

    def test_literal_extended_request_changes_are_bound_to_their_producer_branches(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/'crates/xc-spectral/src/ccm/extended_research.rs').read_text(encoding='utf-8')
        kinds = dict(re.findall(r'"([a-z_]+)"\s*=>\s*"(ccm_[a-z_]+)"',rust_body(source,'pub fn artifact_kind')))
        a=source.index('let mut request = json!(')
        body=source[a:source.index('\n            request\n',a)]
        scopes=[]
        for m in re.finditer(r'if\s+(id\s*==\s*"[a-z_]+"|matches!\(\s*id\s*,\s*(?:"[a-z_]+"\s*\|?\s*)+\))\s*\{',body):
            scopes.append((m.start(),rust_end_brace(body,m.end()-1),set(re.findall(r'"([a-z_]+)"',m.group(1)))))
        checked=0
        for m in re.finditer(r'request\["([a-z_]+)"\]\s*=\s*json!\(\s*("[^"\n]*"|[0-9][0-9_]*)\s*\)',body):
            applicable=set(kinds)
            for start,end,ids in scopes:
                if start<m.start()<end: applicable &= ids
            self.assertTrue(applicable,m.group(0))
            value=json.loads(m.group(2) if m.group(2).startswith('"') else m.group(2).replace('_',''))
            for id in applicable:
                self.assertEqual(impact.CURRENT_REQUEST_PARAMETERS.get(kinds[id],{}).get(m.group(1)),value,(id,m.group(1)))
                checked+=1
        self.assertGreater(checked,100)
        # Both producer and numerical primitive must bind the exact grid epoch.
        distance=(root/'crates/xc-spectral/src/distance.rs').read_text(encoding='utf-8')
        grid=(root/'crates/xc-numerics/src/grid_integral.rs').read_text(encoding='utf-8')
        value=re.search(r'pub const UNIFORM_GRID_SEMANTICS:\s*&str\s*=\s*"([^"]+)"',grid).group(1)
        self.assertEqual(distance.count('"uniform_grid_arithmetic": xc_numerics::grid_integral::UNIFORM_GRID_SEMANTICS'),7)
        for kind in impact.GRID_KINDS:
            self.assertEqual(impact.CURRENT_PARAMETERS[kind]['uniform_grid_arithmetic'],value)

    def test_invalid_algorithm_metadata_is_reported(self):
        artifact(self.root,'a','ccm_sector_eigenvalues',impact.CURRENT_IDENTITIES['ccm_sector_eigenvalues'],algorithm=['not a string'])
        report=impact.inventory([self.root])
        self.assertTrue(report['metadata_errors'])
        self.assertFalse(report['coverage_complete_for_supplied_shards'])


    def test_spectrum_recovery_options_are_bound_without_fixing_the_callers_budget(self):
        root=Path(__file__).resolve().parents[1]
        hp=(root/'crates/xc-spectral/src/ccm/hp.rs').read_text(encoding='utf-8')
        eigen=(root/'crates/xc-numerics/src/eigen.rs').read_text(encoding='utf-8')
        semantics=rust_body(eigen,'pub fn semantics_id(')
        current=impact.CURRENT_PARAMETERS['ccm_sector_spectrum']
        self.assertIn('"'+current['tridiagonal_eigenvector_semantics']+'"',semantics)
        self.assertRegex(hp,r'"tridiagonal_eigenvector_semantics":\s*xc_numerics::eigen::TridiagSolver::BandedInterleaved\.semantics_id\(\)')
        self.assertRegex(hp,r'"eigenvector_iteration_limit":\s*cfg\.inverse_iter_steps')
        self.assertRegex(hp,r'"eigenvector_early_termination":\s*true')
        for limit in [1, 256, 10000]:
            record={'artifact_kind':'ccm_sector_spectrum','resolved_mathematical_parameters':{**current,'eigenvector_iteration_limit':limit}}
            self.assertEqual(impact.additional_identity_mismatches(record),[])
        for missing in ['tridiagonal_eigenvector_semantics','eigenvector_early_termination','eigenvector_iteration_limit']:
            with self.subTest(missing=missing):
                parameters={**current,'eigenvector_iteration_limit':256}
                del parameters[missing]
                artifact(self.root,'a','ccm_sector_spectrum',impact.CURRENT_IDENTITIES['ccm_sector_spectrum'],parameters=parameters)
                self.assertEqual(impact.inventory([self.root])['active_recompute_count'],1)

    def test_algorithm_and_variable_kind_bindings_match_active_producer_fields(self):
        root=Path(__file__).resolve().parents[1]
        hp=(root/'crates/xc-spectral/src/ccm/hp.rs').read_text(encoding='utf-8')
        eigen=(root/'crates/xc-numerics/src/eigen.rs').read_text(encoding='utf-8')
        qr=re.search(r'pub const TRIDIAG_QR_SEMANTICS:\s*&str\s*=\s*"([^"]+)"',eigen).group(1)
        self.assertEqual(impact.QR_SEMANTICS,qr)
        self.assertRegex(hp, r'\.to_owned\(\)\s*\+\s*"\+"\s*\+\s*xc_numerics::eigen::TRIDIAG_QR_SEMANTICS')
        for algorithm in impact.CURRENT_ALGORITHMS['ccm_sector_eigenvalues']:
            prefix=algorithm.removesuffix('+'+qr)
            if prefix!=qr: self.assertIn('"'+prefix+'"',hp)
        roots=rust_body(hp,'fn root_range_semantic_key(')
        self.assertIn('fixed_guard_correction_guards',roots)
        self.assertIn('json!(FIXED_CORRECTION_GUARDS)',roots)
        guards=re.search(r'const FIXED_CORRECTION_GUARDS:\s*\[u32;\s*7\]\s*=\s*\[([^]]+)\]',hp).group(1)
        self.assertEqual(impact.FIXED_CORRECTION_GUARDS,[int(x.strip()) for x in guards.split(',') if x.strip()])
        self.assertIn('{}+confirmed_retained_correction_v2',roots)
        self.assertIn('{}+adaptive_root_precision_v1',roots)
        record=(root/'crates/xc-cache/src/research_artifacts.rs').read_text(encoding='utf-8')
        producer=rust_body(record,'pub fn persist_research_artifact')
        for name in ['CAPTURE_RECEIPT_KIND','HYPOTHESIS_EVALUATION_KIND']:
            kind=re.search(r'pub const '+name+r':\s*&str\s*=\s*"([^"]+)"',record).group(1)
            self.assertIn(name,producer)
            self.assertEqual(impact.CURRENT_IDENTITIES[kind],re.search(r'mathematical_semantics_version:\s*"([^"]+)"',producer).group(1))
        prolate=(root/'crates/xc-spectral/src/prolate.rs').read_text(encoding='utf-8')
        producer=rust_body(prolate,'fn prolate_spectrum_via_cache_model(')
        models=set(re.findall(r'"(prolate-[^"]+)"',producer))
        self.assertEqual(models,{impact.CURRENT_IDENTITIES['prolate_eigenvalue_spectrum']}|impact.ALTERNATE_IDENTITIES['prolate_eigenvalue_spectrum'])
        self.assertIn('mathematical_semantics_version: model.to_owned()',producer)
        self.assertRegex(producer, r'algorithm_semantics:\s*Some\(\s*xc_numerics::eigen::TRIDIAG_QR_SEMANTICS\.to_owned\(\)\s*,?\s*\)')
        self.assertEqual(impact.CURRENT_ALGORITHMS['prolate_eigenvalue_spectrum'],{qr})

    def test_tail_and_band_requests_bind_live_numerical_dependencies(self):
        root=Path(__file__).resolve().parents[1]
        eigen=(root/'crates/xc-numerics/src/eigen.rs').read_text(encoding='utf-8')
        producer=(root/'crates/xc-spectral/src/ccm/extended_research.rs').read_text(encoding='utf-8')
        checkpoint=(root/'crates/xc-spectral/src/ccm/convergence_capture.rs').read_text(encoding='utf-8')
        values={
            'tail_model_checkpoint_arithmetic': 'finite-tail-model-original-matrix-dense-source-recovery-v8',
            'tail_model_householder_arithmetic': re.search(r'pub const STABLE_HOUSEHOLDER_SEMANTICS:\s*&str\s*=\s*"([^"]+)"',eigen).group(1),
            'tail_model_qr_arithmetic': re.search(r'pub const TRIDIAG_QR_SEMANTICS:\s*&str\s*=\s*"([^"]+)"',eigen).group(1),
            'tail_model_vector_recovery_arithmetic': re.search(r'pub const DENSE_EIGENVECTOR_SEMANTICS:\s*&str\s*=\s*"([^"]+)"',(root/'crates/xc-numerics/src/eigen_recovery.rs').read_text(encoding='utf-8')).group(1),
        }
        for field,value in values.items():
            self.assertIn('request["'+field+'"]',producer)
            for kind in ['ccm_tail_operator_analysis','ccm_band_reconstruction']:
                self.assertEqual(impact.CURRENT_REQUEST_PARAMETERS[kind][field],value)
        for live in ['STABLE_HOUSEHOLDER_SEMANTICS','TRIDIAG_QR_SEMANTICS','DENSE_EIGENVECTOR_SEMANTICS']:
            self.assertIn(live,producer)
            self.assertIn(live,checkpoint)
        self.assertIn(values['tail_model_checkpoint_arithmetic'],checkpoint)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "cache-repository.json").write_text(json.dumps({"repository": "synthetic/private-shard"}))

    def tearDown(self):
        self.temp.cleanup()

    def test_only_children_of_defective_certificate_are_flagged(self):
        artifact(self.root, "a", "ccm_tau_matrix", "ordinary-gl-v1")
        artifact(self.root, "b", "ccm_sector_gap_certificate", "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2", ["a"])
        artifact(self.root, "c", "ccm_convergence_diagnostics", "summary-v1", ["b"])
        artifact(self.root, "d", "ccm_certificate_bundle", "ccm-exact-point-source-root-certificate-v0.15.2-v1", ["a"])
        report = impact.inventory([self.root])
        rows = {row["manifest_digest"][0]: row for row in report["artifacts"]}
        self.assertEqual(report["active_affected_count"], 2)
        self.assertEqual(rows["a"]["impact"], "not_flagged_by_this_defect_rule")
        self.assertEqual(rows["d"]["impact"], "not_flagged_by_this_defect_rule")
        self.assertTrue(report["metadata_coverage_complete_for_supplied_shards"])
        self.assertFalse(report["numerical_results_recomputed"])

    def test_unavailable_dependencies_block_complete_coverage(self):
        artifact(self.root, "a", "ccm_tau_matrix", impact.CURRENT_IDENTITIES["ccm_tau_matrix"], ["b"])
        report = impact.inventory([self.root])
        self.assertFalse(report["coverage_complete_for_supplied_shards"])
        self.assertEqual(report["missing_dependency_manifest_digests"], ["b" * 64])

    def test_retired_bad_certificate_is_not_counted_as_active(self):
        artifact(self.root, "a", "ccm_sector_gap_certificate", "ccm-cutoff-free-sector-gap-certificate-v0.14.1-v2", active=False)
        report = impact.inventory([self.root])
        self.assertEqual(report["directly_affected_count"], 1)
        self.assertEqual(report["active_affected_count"], 0)

    def test_unknown_certificate_semantics_require_review(self):
        artifact(self.root, "a", "ccm_sector_gap_certificate", "unrecognized-route")
        self.assertEqual(impact.inventory([self.root])["active_unrecognized_certificate_count"], 1)

    def test_requested_raw_byte_hash_check_detects_mismatch(self):
        artifact(self.root, "a", "ccm_tau_matrix", impact.CURRENT_IDENTITIES["ccm_tau_matrix"])
        report = impact.inventory([self.root], verify_manifest_bytes=True)
        self.assertTrue(report["metadata_errors"])
        self.assertFalse(report["coverage_complete_for_supplied_shards"])

    def test_duplicate_shard_roots_are_rejected(self):
        with self.assertRaises(ValueError):
            impact.inventory([self.root, self.root])

    def test_retained_reduction_replay_epoch_preserves_source_identity(self):
        artifact(self.root, "a", "ccm_even_sector_matrix", impact.CURRENT_IDENTITIES["ccm_even_sector_matrix"])
        artifact(self.root, "b", "ccm_retained_reduction_check", "ccm-retained-reduction-v0.15.0-v1", ["a"])
        artifact(self.root, "c", 'research_capture_receipt', 'managed-research-record-v1', ["b"])
        artifact(self.root, "d", "ccm_retained_reduction_check", impact.CURRENT_IDENTITIES["ccm_retained_reduction_check"], ["a"])
        report = impact.inventory([self.root])
        self.assertEqual(report["metadata_errors"], [])
        self.assertEqual(report["active_recompute_count"], 2)
        self.assertEqual(report["active_affected_count"], 0)
        self.assertNotIn("ccm_even_sector_matrix", report["active_recompute_by_kind"])

    def test_prefix_v5_requests_fresh_exports_without_replacing_matrix_sources(self):
        artifact(self.root, "a", "ccm_even_sector_matrix", impact.CURRENT_IDENTITIES["ccm_even_sector_matrix"])
        artifact(self.root, "b", "ccm_prefix_analysis", "ccm-retained-even-prefix-moments-checked-exports-v1", ["a"])
        artifact(self.root, "c", 'research_capture_receipt', 'managed-research-record-v1', ["b"])
        artifact(self.root, "d", "ccm_prefix_analysis", impact.CURRENT_IDENTITIES["ccm_prefix_analysis"], ["a"])
        artifact(self.root, "e", "ccm_prefix_analysis", "ccm-retained-even-prefix-moments-checked-exports-v2", ["a"])
        artifact(self.root, "f", 'research_capture_receipt', 'managed-research-record-v1', ["e"])
        artifact(self.root, "1", "ccm_prefix_analysis", "ccm-retained-even-prefix-moments-checked-exports-v3", ["a"])
        artifact(self.root, "2", "ccm_prefix_analysis", "ccm-retained-even-prefix-moments-checked-exports-v4", ["a"])
        report = impact.inventory([self.root])
        self.assertEqual(report["metadata_errors"], [])
        self.assertEqual(report["active_recompute_count"], 6)
        self.assertEqual(report["active_affected_count"], 0)


if __name__ == "__main__":
    unittest.main()
