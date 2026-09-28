#!/usr/bin/env python3
"""Generate shared v1 retained-research payload schemas; never edits artifact indexes."""
import json, argparse, copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S={'type':'string'}
SC={'type':'string','pattern':r'^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?![\s\S])'}
D={'type':'string','pattern':r'^[0-9a-f]{64}(?![\s\S])'}
I={'type':'integer','minimum':0}
B={'type':'boolean'}
def nullable(x):return {'anyOf':[x,{'type':'null'}]}
def array(x):return {'type':'array','items':x}
def obj(fields):return {'type':'object','additionalProperties':False,'required':list(fields),'properties':fields}
def enum(*values):return {'enum':list(values)}
KEY=obj(dict(kind=S,logical_key=S,parameters_digest=D))
DEP=obj(dict(key=KEY,content_digest=D,required_quality=enum('validated','cross_checked','certified','published','quarantined','staged','deprecated')))
POINT=obj(dict(ordinal={'type':'integer','minimum':1},value=nullable(SC),source_status=S))
REFERENCE=obj(dict(schema_version={'const':1},definition=S,lambda_squared=SC,precision_bits=I,coefficients=array(SC),approximation_scope=S))
DATASET=obj(dict(schema_version={'const':1},role=enum('known_zeta_ordinates','evaluation_points','retained_ccm_roots'),attribution=S,coordinate={'const':'mellin_t'},precision_bits=I,points=array(POINT)))
OBS=obj(dict(schema_version={'const':1},original_utf8=S,attribution=S,definition=S,hypotheses=array(S),borrowed_inputs=array(S),limitations=array(S)))
TR=obj(dict(ordinal=I,source_status=S,t=nullable(SC),outcome=enum('missing_input','point_measurement','unresolved_derivative','cancellation_limited'),value=nullable(SC),derivative=nullable(SC),sum_absolute_terms=nullable(SC),sum_absolute_derivative_terms=nullable(SC),cancellation_digits=nullable(SC),newton_correction=nullable(SC),newton_remainder=S))
ST=obj(dict(source=D,n_modes=I,value=SC,relative_change=nullable(SC),status=enum('initial','zero_denominator','within_rule','outside_rule')))
KINDS={
 'ccm_reference_source':('prolate',True,obj(dict(spec=REFERENCE,definition_digest=D,basis=S,certification=S))),
 'research_reference_dataset':('ccm-evidence',False,DATASET),
 'research_observation_packet':('ccm-evidence',True,obj(dict(origin={'const':'externally_reported'},validation={'const':'structure_and_bytes_only; numerical_claims_not_replayed'},original_digest=D,observation=OBS))),
 'ccm_reference_projection_analysis':('ccm-distance',True,obj(dict(outcome=enum('normalization_unresolved','rank_or_precision_unresolved','point_measurement'),metric=S,normalization=enum('unit_l2_dx','center_one'),source_center=SC,reference_center=SC,signed_unit_overlap=SC,difference_norm_squared=nullable(SC),gram=array(SC),rhs=array(SC),coefficients=nullable(array(SC)),fit_residual_norm_squared=nullable(SC),minimum_pivot=nullable(SC),fixed_second_component=nullable(SC),b_effective=nullable(SC),b2=nullable(SC),uncertainty=S))),
 'ccm_indexed_transform_analysis':('ccm-evidence',False,obj(dict(coordinate=S,normalization=S,input_role=S,source_precision_bits=I,working_precision_bits=I,finite_support=S,physical_tail=S,second_derivative_cauchy_schwarz_expression=SC,rows=array(TR)))),
 'ccm_operator_energy_analysis':('ccm-evidence',False,obj(dict(lambda_squared=SC,n_modes=I,precision_bits=I,source_eigenvalue=SC,coefficient_norm_squared=SC,rayleigh_quotient=SC,eigenvalue_defect=SC,relative_residual=SC,residual_normalization=enum("absolute_residual_per_coefficient_l2_norm","relative_to_abs_eigenvalue_and_coefficient_l2_norm"),sum_absolute_energy_terms=SC,cancellation_digits=nullable(SC),component_decomposition=S,ground_selection=S))),
 'ccm_root_band_analysis':('ccm-evidence',False,obj(dict(lambda_squared=SC,n_modes=I,source_completeness=S,source_acquisition={"type":"object"},ordinal_scope=S,points=array(POINT),positive_count=I,nonpositive_count=I,missing_count=I,minimum_positive_spacing=nullable(SC),window_inverse_moments=array(SC),omitted_tail=S))),
 'ccm_stabilization_analysis':('ccm-evidence',False,obj(dict(lambda_squared=SC,observable={'const':'serialized_retained_Weil_eigenvalue'},qualification=S,outcome=enum('finite_rule_met','finite_rule_not_met'),rows=array(ST)))),
}

# Extended point diagnostics use named scalar maps; scalar values remain decimal
# strings at the declared precision. No formula-specific target input is embedded.
JET=obj(dict(value=SC,derivative=SC))
RJ=obj(dict(ordinal={'type':'integer','minimum':1},t=SC,reference_window=JET,reference_full=JET,exterior_tail=JET,endpoint_tail_part=nullable(JET),fitted_interior_parts=array(JET),error_normalization=nullable(enum('unit_l2_dx')),source_value_error=nullable(SC),tail_value_error=nullable(SC),source_derivative_error=nullable(SC),root_separation_radius=nullable(SC)))
RJ['properties']['matched_root_ordinal']=nullable({'type':'integer','minimum':1})
RANK=obj(dict(weight=SC,vector=array(SC)))
OP=obj(dict(label=S,source_digest=D,diagonal=array(SC),dense=array(SC),rank_one=array(RANK)))
ATOM=obj(dict(ordinal={'type':'integer','minimum':1},coordinate=SC,weight=SC,family=enum('zero','lattice'),partition=S))
CV=obj(dict(source_digest=D,n_modes=I,precision_bits=I,eigenvalue=SC,coefficients=array(SC),assembly_policy=S))
ALLOW=obj(dict(upper_trial_energy=SC,low_block_lower_bound=SC,high_block_lower_bound=SC,cross_block_norm_bound=SC,hypothesis_record_digest=D,hypotheses=array(S)))
SAMPLED=obj(dict(definition_digest=D,evaluation_policy=S,approximation_scope=S,intervals=I,values=array(SC),basis_values=array(array(SC)),fixed_second_component=nullable(SC),raw_normalizer=SC,trial_coefficients=nullable(array(SC))))
EXTERNAL=obj(dict(schema_version={'const':1},source_eigenpair=D,lambda_squared=SC,n_modes=I,precision_bits=I,convention_id=S,definition_digest=D,approximation_scope=S,target=nullable(SAMPLED),reference_jets=array(RJ),components=array(OP),components_are_complete=B,perturbations=array(OP),deficit=nullable(SC),deficit_kind=nullable(enum('exact_source','fuchs_approximation','other_approximation')),atoms=array(ATOM),atom_coordinate=nullable(S),atom_coverage=nullable(S),tail_checkpoints=array(SC),cluster=array(CV),previous_cluster=array(CV),cluster_boundary_eigenvalues=nullable({'type':'array','items':SC,'minItems':2,'maxItems':2}),energy_allowance=nullable(ALLOW)))
ACTION=obj(dict(label=S,source_digest=D,action=array(SC),convention=S))
TAILFORM=obj(dict(definition_digest=D,dimension=I,finite_zero_form=array(SC),tail_correction=array(SC),lattice_gram=array(SC),tail_operator_error=nullable(SC),coverage=S,hypotheses=array(S)))
TAILFORM['properties']['polynomial_coordinate']=nullable(S)
COMPARISON=obj(dict(source_digest=D,matrix_digest=D,lambda_squared=SC,n_modes=I,precision_bits=I,coefficients=array(SC),matrix=array(SC),eigenvalue=SC,assembly_policy=S))
UNCERTAINTY=obj(dict(unit_state_l2_error=SC,source_certificate_digest=D,hypotheses=array(S)))
RUNONCE=obj(dict(component_actions=array(ACTION),derivative_actions=array(ACTION),log_cutoff_velocity=nullable(SC),reference_vectors=array(array(SC)),comparison=nullable(COMPARISON),tail_form=nullable(TAILFORM),uncertainty=nullable(UNCERTAINTY),producer_notes=array(S)))

RESPONSE=obj(dict(ordinal=I,t=SC,source_digest=D,branch=S,coordinate=S,derivative_parameter=S,activation_convention=S,fixed_velocity=nullable(SC),support_velocity=nullable(SC),total_velocity=nullable(SC)))
SNAPSHOT=obj(dict(state=COMPARISON,selection_policy=S,assembly_policy=S,quadrature_policy=S,root_branch=S,root_coordinate=S,roots=array(POINT)))
BANDATOM=obj(dict(coordinate=SC,signed_weight=SC,family=S))
BAND=obj(dict(degree={'type':'integer','minimum':1,'maximum':2048},coordinate=S,definition_digest=D,atoms=array(BANDATOM),coverage=S,hypotheses=array(S),borrowed_inputs=array(S),input_energy=nullable(SC),scoring_roots=array(SC)))
CONTOUR=obj(dict(left=SC,right=SC,bottom=SC,top=SC,maximum_depth={'type':'integer','minimum':0,'maximum':40},maximum_segments={'type':'integer','minimum':4,'maximum':100000}))
CERT={'type':'object','description':'PortableCcmSectorGapCertificate schema 3; the Toolkit replays full-matrix inertia by its portable schema, historical exact-rational or current directed-MPFR selected-eigenvalue proofs, and all identity checks before deriving any finite source-error allowance. This container schema alone does not validate its proof.','required':['schema_version','lambda_squared','n_modes','cutoff_free_tau','full_matrix_inertia_certificate','even_ground','even_first_excited','odd_ground','parity_invariance_premise','claim_scope'],'properties':{'schema_version':{'enum':[3,4]},'lambda_squared':SC,'n_modes':I}}
COMPLETION=obj(dict(independent_actions=array(ACTION),response_checks=array(RESPONSE),comparisons=array(SNAPSHOT),band=nullable(BAND),contour=nullable(CONTOUR),sector_certificate=nullable(CERT),preparation_notes=array(S)))
RUNONCE['properties']['completion']=COMPLETION

EXTERNAL['properties']['run_once']=RUNONCE
SCALARS={'type':'object','additionalProperties':SC}
EROW=obj(dict(ordinal={'type':'integer','minimum':1},label=S,outcome=enum('point_measurement','certified_finite_enclosure','missing_input','budget_limited','carrier_or_unresolved','unresolved_denominator','cancellation_limited','unresolved_derivative','channels_resolved_budget_unassessed','conditional_budget_met','conditional_budget_not_met','conditional_budget_unresolved'),values=SCALARS,notes=array(S)))
EXTENDED=[
 ('compactness','ccm_compactness_analysis','ccm-evidence',False),
 ('weighted_reference_projection','ccm_weighted_reference_projection','ccm-distance',True),
 ('signed_transform','ccm_signed_transform_analysis','ccm-distance',True),
 ('arithmetic_energy','ccm_arithmetic_energy_analysis','ccm-evidence',True),
 ('directional_response','ccm_directional_response_analysis','ccm-evidence',False),
 ('weighted_tail','ccm_weighted_tail_analysis','ccm-evidence',True),
 ('spectral_cluster','ccm_spectral_cluster_analysis','ccm-evidence',True),
 ('resolution_budget','ccm_resolution_budget_analysis','ccm-evidence',False),
 ('energy_allowance','ccm_energy_allowance_analysis','ccm-evidence',True),
 ('complex_transform','ccm_complex_transform_analysis','ccm-evidence',False),
 ('root_transport','ccm_root_transport_analysis','ccm-evidence',False),
 ('operator_cluster','ccm_operator_cluster_analysis','ccm-evidence',True),
 ('finite_section_transfer','ccm_finite_section_transfer','ccm-evidence',False),
 ('tail_operator','ccm_tail_operator_analysis','ccm-evidence',True),
 ('observable_budget','ccm_observable_budget_analysis','ccm-evidence',False),
 ('capture_preflight','ccm_capture_preflight','ccm-evidence',False),
 ('consistency','ccm_consistency_analysis','ccm-evidence',False),
 ('configuration_comparison','ccm_configuration_comparison','ccm-evidence',True),
 ('band_reconstruction','ccm_band_reconstruction','ccm-evidence',True),
 ('transform_enclosure','ccm_transform_enclosure','ccm-evidence',False),
]
for diagnostic,kind,family,private in EXTENDED:
 KINDS[kind]=(family,private,obj(dict(diagnostic={'const':diagnostic},outcome=enum('point_measurement','certified_finite_enclosure','partial_unresolved','unresolved','missing_input','rank_or_precision_unresolved','conditional_bound_expression','sufficient_bound_unavailable'),reason=nullable(S),lambda_squared=SC,n_modes=I,source_precision_bits=I,working_precision_bits=I,convention=S,assurance={'const':'point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'},values=SCALARS,rows=array(EROW))))
KINDS['ccm_external_research_source']=('prolate',True,EXTERNAL)
KINDS['ccm_transform_enclosure'][2]['properties']['assurance']={'const':'finite_retained_function_enclosures; source_scope_explicit; no_infinite_limit_claim'}


PROJECTION=obj(dict(working_precision_bits=I,normalization=enum('unit_l2_dx','center_one'),fixed_second_component=nullable(SC)))
RESEARCH_INPUT=obj(dict(schema_version={'const':1},reference=REFERENCE,basis=array(REFERENCE),projection=PROJECTION))
TAILRECIPE=obj(dict(basis_polynomials=array(array(SC)),tail_correction=nullable(array(SC)),hypotheses=array(S)))
CHUNK=obj(dict(relative_path=S,sha256=D,bytes={'type':'integer','minimum':1},rows={'type':'integer','minimum':1}))
ATOMKEY=obj(dict(family=S,partition=S,ordinal={'type':'integer','minimum':1}))
ATOMEVAL=obj(dict(ordinal={'type':'integer','minimum':1},coordinate=SC,label=S,exclude=array(ATOMKEY)))
ATOMEVAL['required']=['ordinal','coordinate','label']
ATOMPOLICY=obj(dict(maximum_atoms={'type':'integer','minimum':1},maximum_input_bytes={'type':'integer','minimum':1},weighted_chunks=array(CHUNK),band_chunks=array(CHUNK),evaluations=array(ATOMEVAL),cutoffs={**array(SC),'maxItems':32},tail_recipe=nullable(TAILRECIPE)))
ATOMPOLICY['required']=['maximum_atoms','maximum_input_bytes']
COMPLETION['properties']['atom_analysis']=nullable(ATOMPOLICY)
PREPARATION=obj(dict(schema_version={'const':1},finite_reference=nullable(RESEARCH_INPUT),sampled_reference=nullable(SAMPLED),lambda_squared=SC,precision_bits=I,definition_digest=D,approximation_scope=S,weighted_atoms=array(ATOM),atom_coordinate=nullable(S),atom_coverage=nullable(S),tail_form=nullable(TAILFORM),completion=nullable(COMPLETION)))
PREPARATION['properties']['tail_recipe']=nullable(TAILRECIPE)
PREPARATION['required']=['schema_version','lambda_squared','precision_bits','definition_digest','approximation_scope']

# Conditional requirements preserve qualified absence and older producer revisions.
OPTIONS=obj(dict(working_precision_bits=I,maximum_rows=I,maximum_directional_rows=I,maximum_estimated_output_bytes=I,exponential_rates=array(SC),relative_tolerance=SC))
OPTIONS['properties']['maximum_working_bytes']=nullable(I)
CORE={
 'compactness':['transform_origin','transform_second_derivative','transform_fourth_derivative'],
 'weighted_reference_projection':['source_raw_center','weighted_l1','weighted_l2_squared','signed_integral'],
 'arithmetic_energy':['retained_weil_energy','total_tau_energy'],
 'spectral_cluster':['minimum_gram_pivot','source_cluster_leakage_squared'],
 'resolution_budget':['conditional_contiguous_prefix','relative_tolerance','finite_curvature_expression'],
 'energy_allowance':['upper_trial_energy','low_block_lower_bound','high_block_lower_bound','cross_block_norm_bound','denominator'],
 'complex_transform':['normalization_anchor'],
 'operator_cluster':['subspace_dimension','retained_energy_shift','source_leakage_squared'],
 'tail_operator':['model_energy','retained_energy_for_scoring_only'],
 'observable_budget':['transform_origin','origin_absolute_terms'],
 'capture_preflight':['maximum_output_bytes','maximum_working_bytes','root_count'],
 'band_reconstruction':['signed_mass'],
 'transform_enclosure':['contour_left','contour_right','contour_top','contour_bottom','origin_real_lower','origin_real_upper','unresolved_segments'],
}
ROWCORE={
 'compactness':['rate','analytic_integral','sum_absolute_terms'],
 'signed_transform':['t','value_actual','derivative_actual','value_reference_closure_defect','derivative_reference_closure_defect'],
 'arithmetic_energy':['energy'],
 'directional_response':['t','tau','direction_norm_squared','directional_energy'],
 'spectral_cluster':['eigenvalue'],
 'resolution_budget':['t','transform','derivative'],
 'complex_transform':['z_re','z_im','value_re','value_im','derivative_re','derivative_im'],
 'root_transport':['t','tau','direction_norm_squared','directional_energy'],
 'operator_cluster':['row','column','compressed_operator','coupling_gram'],
 'finite_section_transfer':['n_modes','retained_mass','omitted_mass','truncated_energy'],
 'observable_budget':['t','value','derivative'],
 'capture_preflight':['available'],
 'consistency':['action_difference_norm','signed_energy_difference'],
 'configuration_comparison':['comparison_C','comparison_N','comparison_P','signed_energy_difference'],
}
def measurement_contract(schema, diagnostic):
 request=obj(dict(semantics=enum('extended-retained-diagnostics-v2','extended-retained-diagnostics-v3','extended-retained-diagnostics-v4'),diagnostic={'const':diagnostic},expected_rows=nullable(I),options=OPTIONS,external_input_digest=nullable(D),state_selection_policy=S,state_manifest_tags={'type':'object','additionalProperties':S}))
 if diagnostic in ('band_reconstruction','transform_enclosure'):
  # Optional for historical packets; current producers always bind capability.
  request['properties']['arb_available']={'type':'boolean'}
 if diagnostic=='band_reconstruction':
  request['properties'].update(basis_disk_budget_bytes=I,checkpoint_block_budget_bytes=I)
 if diagnostic=='transform_enclosure':
  request['properties']['semantics']=enum('extended-retained-diagnostics-v2','extended-retained-diagnostics-v3','extended-retained-diagnostics-v4','extended-retained-diagnostics-v4-source-error-hull')
  request['properties']['enclosure_algorithm']=enum('centered-taylor-48-integral-remainder-v1','centered-taylor-48-integral-remainder-v2','centered-taylor-48-integral-remainder-v3-stored-points')
 if diagnostic=='energy_allowance':
  request['properties']['allowance_interpretation']={'const':'trial-energy-scale-v1'}
 schema['properties']['request']=request
 data=schema['properties']['data']
 resolved=['point_measurement','certified_finite_enclosure','conditional_bound_expression']
 data['allOf']=[{'if':{'properties':{'outcome':enum(*resolved)}},'then':{'properties':{'values':{'required':CORE.get(diagnostic,[])}}}}]
 if diagnostic=='energy_allowance':
  # Preserve old packets; require scale information from the new producer.
  schema.setdefault('allOf',[]).append({'if':{'properties':{'request':{'required':['allowance_interpretation']},'data':{'properties':{'outcome':{'const':'conditional_bound_expression'}}}}},'then':{'properties':{'data':{'properties':{'reason':S,'values':{'required':['conditional_energy_allowance','conditional_vector_allowance','trial_energy_magnitude']}}}}}})
 # Keep scalar maps extensible for declared component labels and indexed matrices.
 rows=data['properties']['rows']['items']=json.loads(json.dumps(EROW))
 if diagnostic!='transform_enclosure':
  data['properties']['outcome']['enum']=[v for v in data['properties']['outcome']['enum'] if v!='certified_finite_enclosure']
  rows['properties']['outcome']['enum']=[v for v in rows['properties']['outcome']['enum'] if v!='certified_finite_enclosure']
 if diagnostic in ROWCORE:
  rows['allOf']=[{'if':{'properties':{'outcome':enum('point_measurement','certified_finite_enclosure')}},'then':{'properties':{'values':{'required':ROWCORE[diagnostic]}}}}]
 bylabel={
 'band_reconstruction':{'signed_functional_stieltjes_recurrence':['jacobi_diagonal','next_signed_norm_squared','reorthogonalization_correction'],'band_cutoff':['zero_atom_cutoff','first_model_band_root','last_model_band_root']},
 'tail_operator':{'generalized_model_eigenvalue':['energy'],'tail_model_cutoff':['zero_atom_cutoff','model_energy']},
 'weighted_tail':{'signed_atom_kernel':['signed_kernel_1','signed_kernel_2','signed_kernel_3','used_atoms','explicitly_excluded_atoms']},
 'transform_enclosure':{'contour_segment':['start_re','start_im','end_re','end_im','argument_increment_lower','argument_increment_upper'],'retained_root_enclosure':['input_ordinal','value_real_lower','value_real_upper','derivative_real_lower','derivative_real_upper']}}
 for label,keys in bylabel.get(diagnostic,{}).items():
  rows.setdefault('allOf',[]).append({'if':{'properties':{'label':{'const':label},'outcome':enum('point_measurement','certified_finite_enclosure')}},'then':{'properties':{'values':{'required':keys}}}})
 if diagnostic=='signed_transform':
  schema.setdefault('allOf',[]).append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[8:]),'data':{'properties':{'outcome':{'const':'point_measurement'}}}}},'then':{'properties':{'data':{'properties':{'values':{'required':['arithmetic_precision_bits','normalization_precision_bits']}}}}}})
 if diagnostic=='weighted_tail':
  rows.setdefault('allOf',[]).append({'if':{'properties':{'label':{'not':{'const':'signed_atom_kernel'}},'outcome':{'const':'point_measurement'}}},'then':{'properties':{'values':{'required':['cutoff','included_mass','included_absolute_mass','included_count','remaining_supplied_mass','weighted_inverse_moment_1','weighted_inverse_moment_2','weighted_inverse_moment_3']}}}})
 # Newer band/tail measurement revisions are distinguishable from original v2.
 extra={'band_reconstruction':['band_inverse_moment_one','band_inverse_moment_two','band_inverse_moment_three','band_inverse_moment_root_count'],'tail_operator':['model_energy_without_tail','tail_energy_lift']}.get(diagnostic)
 if extra:
  schema.setdefault('allOf',[]).append({'if':{'properties':{'request':{'properties':{'semantics':enum('extended-retained-diagnostics-v3','extended-retained-diagnostics-v4')}},'data':{'properties':{'outcome':{'const':'point_measurement'}}}}},'then':{'properties':{'data':{'properties':{'values':{'required':extra}}}}}})

BESPOKE_REQUESTS={
 'ccm_reference_source':REFERENCE,
 'research_reference_dataset':DATASET,
 'research_observation_packet':OBS,
 'ccm_external_research_source':obj(dict(input_digest=D)),
 'ccm_reference_projection_analysis':obj(dict(options=PROJECTION,reference=D,ordered_basis=array(D),metric=S)),
 'ccm_operator_energy_analysis':obj(dict(precision_bits=I,formula=S)),
 'ccm_root_band_analysis':obj(dict(precision_bits=I,rule=S)),
 'ccm_stabilization_analysis':obj(dict(options=obj(dict(working_precision_bits=I,relative_tolerance=SC,consecutive_steps=I)),ordered_sources=array(D))),
 'ccm_indexed_transform_analysis':obj(dict(options=obj(dict(working_precision_bits=I,maximum_rows=I,maximum_estimated_output_bytes=I)),dataset=DATASET,formula=S)),
}

# Explicit archived revisions; unknown future semantics must fail validation.
RETAINED_REVISIONS=tuple('ccm-retained-research-observations-v'+str(i) for i in range(1,23))

def revision_contract(schema,kind):
 schema['properties']['semantics']=enum(*RETAINED_REVISIONS)
 data=schema['properties']['data'];request=schema['properties']['request']
 def require_from(first,*,data_fields=(),request_fields=()):
  properties={}
  if data_fields:properties['data']={'required':list(data_fields)}
  if request_fields:properties['request']={'required':list(request_fields)}
  schema.setdefault('allOf',[]).append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[first-1:])},'required':['semantics']},'then':{'properties':properties}})
 if kind=='ccm_reference_projection_analysis':
  data['properties'].update(pivot_metric=S,arithmetic_precision_bits={'type':'integer','minimum':128,'maximum':1004096},arithmetic_enclosures={'type':'object','additionalProperties':{**array(SC),'minItems':2,'maxItems':2},'required':['source_center','reference_center','signed_unit_overlap']})
  request['properties'].update(projection_arithmetic={'const':'stable_normalization_interval_gram_v1'},maximum_additional_guard_bits={'const':4096},projection_output={'const':'midpoints_with_outward_decimal_enclosures_v1'})
  require_from(3,data_fields=['pivot_metric'])
  require_from(8,data_fields=['arithmetic_precision_bits','arithmetic_enclosures'],request_fields=['projection_arithmetic','maximum_additional_guard_bits','projection_output'])
  schema.setdefault('allOf',[]).append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[7:])}},'then':{'properties':{'data':{'properties':{'uncertainty':{'const':'finite_stored_inputs_with_arithmetic_enclosures; no_reference_approximation_or_construction_error_enclosure'}}}}}})
  schema.setdefault('allOf',[]).append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[7:]),'data':{'properties':{'outcome':enum('point_measurement','rank_or_precision_unresolved')}}}},'then':{'properties':{'data':{'properties':{'arithmetic_enclosures':{'required':['difference_norm_squared']}}}}}})
  schema['allOf'].append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[7:]),'data':{'properties':{'outcome':{'const':'point_measurement'}}}}},'then':{'properties':{'data':{'properties':{'arithmetic_enclosures':{'required':['fit_residual_norm_squared','minimum_pivot']}}}}}})
 if kind=='ccm_indexed_transform_analysis':
  props=dict(transform_arithmetic=enum('exact_cutoff_stored_points_directed_sinc_v1','exact_cutoff_stored_points_directed_sinc_root_convention_v2'),maximum_transform_guard_bits={'const':4096},curvature_output={'const':'outward_upper_endpoint_v1'})
  request['properties'].update(props);require_from(13,request_fields=props)
 if kind=='ccm_stabilization_analysis':
  props=dict(acceptance_rounding={'const':'exact_decimal_down'},relative_change_arithmetic={'const':'exact_stored_points_outward_upper_v2'})
  request['properties'].update(props);require_from(22,request_fields=props)
 if kind=='ccm_operator_energy_analysis':
  data['required'].remove('residual_normalization')
  request['properties']['residual_semantics']={'const':'raw_matrix_explicit_zero_eigenvalue_normalization_v2'}
  schema.setdefault('allOf',[]).append({'if':{'properties':{'request':{'required':['residual_semantics']}}},'then':{'properties':{'data':{'required':['residual_normalization']}}}})
 diagnostic=next((diagnostic for diagnostic,k,_,_ in EXTENDED if k==kind),None)
 if diagnostic is None:return
 request['properties']['resource_admission']={'const':'resolved_working_bytes_v1'}
 request['properties']['duplicate_coordinate_policy']={'const':'all_ambiguous_members_withheld_v2'}
 request['properties']['ladder_positivity_policy']={'const':'all_required_recurrence_steps_including_failed_v2'}
 if diagnostic in ('observable_budget','tail_operator'):
  request['properties']['declared_error_semantics']={'const':'exact_decimal_upper_bound_v2'}

 if diagnostic=='complex_transform':
  request['properties'].update(maximum_parallel_rows={'type':'integer','minimum':1,'maximum':4096},workspace_admission={'const':'configured_row_block_bound_v1'})
 if diagnostic=='band_reconstruction':
  request['properties']['exact_contraction_admission']={'const':'all_block_workspace_bound_v1'}

 request['properties']['source_unit_arithmetic']=enum('binary_scaled_hypot_checked_v1','binary_scaled_hypot_checked_range_v2')
 require_from(4,request_fields=['source_unit_arithmetic'])
 def revised_parameter(name,old,new,*,additional_current=()):
  older=old if isinstance(old,list) else [old]
  request['properties'][name]=enum(*older,new,*additional_current)
  current=enum(new,*additional_current) if additional_current else {'const':new}
  schema.setdefault('allOf',[]).append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[21:])},'required':['semantics']},'then':{'properties':{'request':{'required':[name],'properties':{name:current}}}},'else':{'properties':{'request':{'properties':{name:enum(*older)}}}}})
 revised_parameter('source_unit_arithmetic','binary_scaled_hypot_checked_v1','binary_scaled_hypot_checked_range_v2')

 fields={
  'finite_section_transfer':(15,dict(finite_transfer_arithmetic={'const':'original_points_scaled_prefix_shifted_residual_v1'},finite_transfer_output={'const':'midpoints_with_outward_decimal_enclosures_v1'},maximum_finite_transfer_guard_bits={'const':4096})),
  'consistency':(15,dict(consistency_arithmetic={'const':'original_points_scaled_action_difference_v1'},consistency_output={'const':'midpoints_with_outward_decimal_enclosures_v1'},maximum_consistency_guard_bits={'const':4096})),
  'operator_cluster':(15,dict(cluster_basis_arithmetic={'const':'original_points_unit_columns_before_rank_threshold_v1'})),
  'observable_budget':(14,dict(observation_arithmetic={'const':'original_points_directed_l2_transport_v1'},observation_output={'const':'allowance_upper_margin_lower_with_expression_enclosures_v1'})),
  'configuration_comparison':(14,dict(comparison_arithmetic={'const':'original_points_scaled_overlap_shifted_residual_v1'},comparison_output={'const':'midpoints_with_outward_decimal_enclosures_v1'},maximum_comparison_guard_bits={'const':4096})),
  'energy_allowance':(12,dict(allowance_arithmetic={'const':'declared_points_separate_binary_scales_intervals_v1'},maximum_allowance_guard_bits={'const':4096},allowance_output={'const':'midpoints_with_outward_decimal_enclosures_v1'})),
  'directional_response':(11,dict(directional_arithmetic={'enum':['stored_points_homogeneous_projected_resolvent_intervals_v1','stored_points_projected_resolvent_displacement_checked_v2']},maximum_directional_guard_bits={'const':4096},directional_output={'const':'midpoints_with_outward_decimal_enclosures_v1'},root_point_precision={'const':'declared_payload_precision_v1'})),
  'arithmetic_energy':(10,dict(energy_arithmetic=enum('stored_points_scaled_quadratic_intervals_v1','stored_points_scaled_quadratic_intervals_deficit_kind_v2'),maximum_energy_guard_bits={'const':4096},energy_output={'const':'midpoints_with_outward_decimal_enclosures_v1'})),
  'signed_transform':(9,dict(signed_channel_arithmetic={'const':'exact_center_declared_points_single_round_channels_v2'})),
  'tail_operator':(2,dict(l2_normalization_arithmetic={'const':'power_two_scaled_max_precision_l2_v2'})),
  'compactness':(4,dict(compactness_arithmetic=enum('directed_enclosure_agreed_rounding_v1','directed_enclosure_agreed_rounding_or_unresolved_v2'),maximum_additional_guard_bits={'const':4096})),
  'weighted_tail':(5,dict(atom_arithmetic={'const':'stored_points_exact_mass_directed_moments_v2'},maximum_atom_guard_bits={'const':4096},maximum_atom_exponent_span_bits={'const':1000000})),
  'spectral_cluster':(6,dict(cluster_arithmetic={'const':'stored_points_scaled_unit_checked_gram_v3'},maximum_cluster_guard_bits={'const':4096},cluster_precision_policy={'const':'unit_column_pivot_proxy_v1'})),
  'weighted_reference_projection':(7,dict(weighted_profile_arithmetic=enum('stored_points_combined_difference_interval_gram_v1','stored_points_combined_difference_interval_gram_unresolved_v2'),maximum_weighted_profile_guard_bits={'const':4096},weighted_profile_output={'const':'midpoint_with_outward_decimal_enclosures_v1'})),
 }
 if diagnostic in fields:
  first,properties=fields[diagnostic];request['properties'].update(properties);require_from(first,request_fields=properties)
 if diagnostic=='signed_transform':
  revised_parameter('signed_channel_arithmetic','exact_center_declared_points_single_round_channels_v2','exact_cutoff_center_declared_points_checked_channels_v3')
 if diagnostic in ('signed_transform','resolution_budget','observable_budget','configuration_comparison'):
  props=dict(transform_arithmetic={'const':'exact_cutoff_stored_points_directed_sinc_v1'},maximum_transform_guard_bits={'const':4096})
  request['properties'].update(props);require_from(13,request_fields=props)
 if diagnostic=='root_transport':
  props=dict(transport_arithmetic={'const':'exact_cutoff_stored_points_directional_intervals_v1'},transport_output={'const':'midpoints_with_outward_decimal_enclosures_v1'},maximum_transport_guard_bits={'const':4096})
  request['properties'].update(props);require_from(19,request_fields=props)
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise';new='finite_stored_point_arithmetic_enclosures; inherited directional intervals; conditional root/displacement hypotheses; no source-error or root certificate'
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[18:])},'required':['semantics']}
  checks=[]
  for name in ['component_forcing_sum','absolute_component_forcing_sum','forcing_closure_defect','support_motion','operator_motion','conditional_total_physical_velocity','retained_secular_pole_motion','retained_total_velocity','retained_transport_additivity_defect']:
   checks.append({'if':{'properties':{'values':{'required':[name]}}},'then':{'properties':{'values':{'required':[name+'_lower',name+'_upper','transport_arithmetic_precision_bits']}}}})
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new},'rows':{'items':{'allOf':checks}}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
 if diagnostic=='transform_enclosure':
  props=dict(enclosure_algorithm=enum('centered-taylor-48-integral-remainder-v3-stored-points','centered-taylor-48-integral-remainder-minus-fourier-v4-stored-points'),enclosure_point_precision={'const':'declared_payload_and_external_precision_v1'},enclosure_decimal_output={'const':'outward_endpoints_v1'})
  revised_parameter('semantics',['extended-retained-diagnostics-v2','extended-retained-diagnostics-v3','extended-retained-diagnostics-v4'],'extended-retained-diagnostics-v4-source-error-hull',additional_current=('extended-retained-diagnostics-v5-minus-fourier-source-error-hull',))
  request['properties']['enclosure_algorithm']['enum'].append('centered-taylor-48-integral-remainder-minus-fourier-v4-stored-points')
  request['properties']['enclosure_fourier_semantics']={'const':'retained_fourier_minus_sign_at_original_coordinates_v1'}
  minus=dict(semantics={'const':'extended-retained-diagnostics-v5-minus-fourier-source-error-hull'},enclosure_algorithm={'const':'centered-taylor-48-integral-remainder-minus-fourier-v4-stored-points'},enclosure_fourier_semantics={'const':'retained_fourier_minus_sign_at_original_coordinates_v1'})
  changed={'anyOf':[{'required':[name],'properties':{name:constraint}} for name,constraint in minus.items()]}
  schema.setdefault('allOf',[]).append({'if':{'properties':{'request':changed}},'then':{'properties':{'request':{'required':list(minus),'properties':minus}}}})
  # The enum above retains earlier algorithms; require the new identifiers by revision.
  request['properties'].update({k:v for k,v in props.items() if k!='enclosure_algorithm'});require_from(19,request_fields=props)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[18:])},'required':['semantics']}
  root={'if':{'properties':{'label':{'const':'retained_root_enclosure'},'values':{'required':['value_real_lower']}}},'then':{'properties':{'values':{'required':['t','t_lower','t_upper','input_point_precision_bits']}}}}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'request':{'properties':props},'data':{'properties':{'rows':{'items':{'allOf':[root]}}}}}}})
 if diagnostic=='band_reconstruction':
  props=dict(signed_band_arithmetic={'const':'declared_points_normalized_recurrence_exact_contractions_v1'},maximum_signed_band_exact_bits={'const':8000000},signed_band_inverse_arithmetic={'const':'relative_zero_guard_scaled_directed_sums_v1'},maximum_signed_band_inverse_guard_bits={'const':4096})
  request['properties'].update(props);require_from(20,request_fields=props)
 if diagnostic=='band_reconstruction':
  props=dict(polynomial_root_window={'const':'common_binary_scale_exact_rational_cauchy_v2'})
  request['properties'].update(props);require_from(22,request_fields=props)
 if diagnostic=='band_reconstruction':
  props=dict(polynomial_band_arithmetic={'const':'stored_polynomial_exact_newton_inverse_moments_v1'},polynomial_root_output={'const':'outward_root_bounds_and_safe_midpoints_v1'},maximum_polynomial_band_exact_bits={'const':8000000})
  request['properties'].update(props);require_from(21,request_fields=props)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[20:])},'required':['semantics']}
  root={'if':{'properties':{'label':{'const':'arithmetic_model_polynomial_root'}}},'then':{'properties':{'values':{'required':['model_band_root','rounded_polynomial_root_lower','rounded_polynomial_root_upper']}}}}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'rows':{'items':{'allOf':[root]}}}}}}})
 if diagnostic=='tail_operator':
  # The arithmetic stamp extends v22 request identities; archived v22 reports
  # without it remain readable and are not upgraded to this arithmetic.
  request['properties']['model_linear_algebra_arithmetic']={'const':'exact_stored_dot_product_stages_and_tail_bound_v0.15.2-v2'}
 if diagnostic in ('tail_operator','band_reconstruction'):
  props=dict(tail_model_arithmetic={'const':'declared_points_exact_dyadic_recipe_forms_v2'},maximum_tail_form_exact_bits={'const':8000000})
  request['properties'].update(props);require_from(18,request_fields=props)
  # Earlier packets did not carry numerical-kernel dependency stamps. New
  # checkpoint-stamped packets must bind every kernel used to recover vectors.
  dependencies=dict(tail_model_checkpoint_arithmetic={'const':'finite-tail-model-original-matrix-dense-source-recovery-v8'},tail_model_householder_arithmetic={'const':'householder-scaled-opposite-sign-v1'},tail_model_qr_arithmetic={'const':'tridiag-qr-working-unit-deflation-exponent-safe-hypot-v3'},tail_model_vector_recovery_arithmetic={'const':'dense-eigenvector-requested-source-rounding-exact-count-scaling-directed-index-gap-angle-v3'})
  request['properties'].update(dependencies)
  schema.setdefault('allOf',[]).append({'if':{'properties':{'request':{'required':['tail_model_checkpoint_arithmetic']}}},'then':{'properties':{'request':{'required':list(dependencies)}}}})
 if diagnostic=='complex_transform':
  props=dict(complex_arithmetic=enum('exact_cutoff_original_points_directed_entire_sinc_v1','exact_cutoff_original_points_directed_entire_minus_sinc_v2'),complex_point_construction=enum('original_probes_exact_affine_contour_v1','original_probes_exact_affine_contour_minus_fourier_v2'),complex_output={'const':'midpoints_with_outward_decimal_enclosures_v1'},maximum_complex_guard_bits={'const':4096},complex_root_point_precision={'const':'declared_payload_precision_v1'})
  request['properties'].update(props);require_from(17,request_fields=props)
  request['properties']['complex_fourier_semantics']={'const':'retained_fourier_minus_sign_at_original_coordinates_v1'}
  minus=dict(complex_arithmetic={'const':'exact_cutoff_original_points_directed_entire_minus_sinc_v2'},complex_point_construction={'const':'original_probes_exact_affine_contour_minus_fourier_v2'},complex_fourier_semantics={'const':'retained_fourier_minus_sign_at_original_coordinates_v1'})
  changed={'anyOf':[{'required':[name],'properties':{name:constraint}} for name,constraint in minus.items()]}
  schema.setdefault('allOf',[]).append({'if':{'properties':{'request':changed}},'then':{'properties':{'request':{'required':list(minus),'properties':minus}}}})
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new='finite_stored_point_arithmetic_enclosures; exact cutoff and retained ordinates; samples do not certify contour counts, roots or source errors'
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[16:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  def bounded(names):return names+[name+'_'+side for name in names for side in ['lower','upper']]+['arithmetic_precision_bits']
  core=bounded(['input_ordinal','z_re','z_im','value_re','value_im','derivative_re','derivative_im','sum_absolute_terms','normalization_denominator_resolved','log_derivative_denominator_resolved'])
  measured={'anyOf':[{'properties':{'values':{'required':['value_re']}}},{'properties':{'outcome':enum('point_measurement','unresolved_denominator')}}]}
  checks=[{'if':measured,'then':{'properties':{'values':{'required':core}}}}]
  for name in ['normalized','log_derivative']:
   checks.append({'if':{'properties':{'values':{'required':[name+'_re']}}},'then':{'properties':{'values':{'required':bounded([name+'_re',name+'_im'])}}}})
  checks.append({'if':{'properties':{'outcome':{'const':'point_measurement'}}},'then':{'properties':{'values':{'required':bounded(['normalized_re','normalized_im','log_derivative_re','log_derivative_im'])}}}})
  present={'properties':{'outcome':enum('point_measurement','partial_unresolved')}}
  schema['allOf'].append({'if':current,'then':{'properties':{'data':{'allOf':[{'if':present,'then':{'properties':{'values':{'required':bounded(['normalization_anchor'])},'rows':{'items':{'allOf':checks}}}}}]}}}})
 if diagnostic=='operator_cluster':
  props=dict(cluster_operator_arithmetic={'const':'original_points_shift_before_interval_projection_lu_v1'},cluster_operator_output={'const':'midpoints_with_outward_decimal_enclosures_v1'},maximum_cluster_operator_guard_bits={'const':4096})
  request['properties'].update(props);require_from(16,request_fields=props)
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new='finite_stored_point_arithmetic_enclosures; numerical selected subspace only; no full-declared-span, source-error or spectral-selection certificate'
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[15:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  def bounded(names):return names+[name+'_'+side for name in names for side in ['lower','upper']]+['arithmetic_precision_bits']
  values=bounded(['subspace_dimension','retained_energy_shift','source_leakage_squared','estimated_factorization_workspace_bytes','column_selection_threshold','discarded_reference_columns'])
  core=bounded(['row','column','compressed_operator','coupling_gram'])
  solved=bounded(['signed_complement_feedback','effective_operator','solve_relative_residual'])
  measured={'properties':{'outcome':enum('point_measurement','partial_unresolved')}}
  row_checks=[{'properties':{'values':{'required':core}}},{'if':{'properties':{'outcome':{'const':'point_measurement'}}},'then':{'properties':{'values':{'required':solved}}}}]
  schema['allOf'].append({'if':current,'then':{'properties':{'data':{'allOf':[{'if':measured,'then':{'properties':{'values':{'required':values},'rows':{'items':{'allOf':row_checks}}}}}]}}}})
 if diagnostic in ('finite_section_transfer','consistency'):
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new={'finite_section_transfer':'finite_stored_point_arithmetic_enclosures; projected retained state only; external comparison source not certified; no convergence claim','consistency':'finite_stored_point_arithmetic_enclosures; common signed unit-state convention and source independence remain external premises'}[diagnostic]
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[14:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  def bounded(names):return names+[name+'_'+side for name in names for side in ['lower','upper']]
  core=['n_modes','retained_mass','omitted_mass','low_residual_squared','high_forcing_squared','truncated_energy'] if diagnostic=='finite_section_transfer' else ['action_difference_norm','signed_energy_difference']
  measured={'anyOf':[{'properties':{'values':{'required':[core[0]]}}},{'properties':{'outcome':{'const':'point_measurement'}}}]}
  row_checks=[{'if':measured,'then':{'properties':{'values':{'required':bounded(core)+['arithmetic_precision_bits']}}}}]
  extra={'properties':{'rows':{'items':{'allOf':row_checks}}}}
  if diagnostic=='finite_section_transfer':
   extra['allOf']=[{'if':{'properties':{'outcome':enum('point_measurement','partial_unresolved')}},'then':{'properties':{'values':{'required':['arithmetic_precision_bits']}}}}, {'if':{'properties':{'values':{'required':['comparison_block_frobenius_difference']}}},'then':{'properties':{'values':{'required':bounded(['comparison_block_frobenius_difference','comparison_precision_bits'])}}}}]
  schema['allOf'].append({'if':current,'then':{'properties':{'data':extra}}})
 if diagnostic in ('observable_budget','configuration_comparison'):
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new={'observable_budget':'finite_stored_point_arithmetic_enclosures; conditional finite-support L2 transport; external state-error premise not certified','configuration_comparison':'finite_stored_point_arithmetic_enclosures; external comparison source and selection premises not certified; no convergence claim'}[diagnostic]
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[13:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  def bounded(names):return names+[name+'_'+side for name in names for side in ['lower','upper']]
  row_core=['t','value','derivative','absolute_value_terms','absolute_derivative_terms'] if diagnostic=='observable_budget' else ['comparison_C','comparison_N','comparison_P','signed_energy_difference']
  measured={'anyOf':[{'properties':{'values':{'required':[row_core[0]]}}},{'properties':{'outcome':enum('point_measurement','channels_resolved_budget_unassessed')}}]}
  row_checks=[{'if':measured,'then':{'properties':{'values':{'required':bounded(row_core)+['arithmetic_precision_bits']}}}}]
  if diagnostic=='observable_budget':
   row_checks.append({'if':{'properties':{'values':{'required':['declared_unit_state_l2_error']}}},'then':{'properties':{'values':{'required':bounded(['declared_unit_state_l2_error','conditional_value_error','conditional_derivative_error','conditional_slope_lower_margin'])}}}})
  rows={'items':{'allOf':row_checks}}
  extra={'properties':{'rows':rows}}
  if diagnostic=='observable_budget':
   origin={'if':{'properties':{'outcome':enum('point_measurement','partial_unresolved')}},'then':{'properties':{'values':{'required':bounded(['transform_origin','origin_absolute_terms'])+['arithmetic_precision_bits']}}}}
   errors={'if':{'properties':{'values':{'required':['declared_unit_state_l2_error']}}},'then':{'properties':{'values':{'required':bounded(['declared_unit_state_l2_error','declared_origin_error','conditional_origin_lower_margin'])}}}}
   extra['allOf']=[origin,errors]
  schema['allOf'].append({'if':current,'then':{'properties':{'data':extra}}})
 if diagnostic=='resolution_budget':
  props=dict(resolution_arithmetic={'const':'original_point_intervals_conditional_distance_v1'},resolution_output={'const':'outward_allowance_upper_endpoints_with_expression_enclosures_v1'})
  request['properties'].update(props);require_from(13,request_fields=props)
  revised_parameter('resolution_arithmetic','original_point_intervals_conditional_distance_v1','exact_decimal_tolerance_original_point_conditional_distance_v2')
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new='finite_stored_point_arithmetic_enclosures; conditional root distance bounds; external source errors and target isolation or curvature hypotheses not certified'
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[12:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  def bounded(names):return names+[name+'_'+side for name in names for side in ['lower','upper']]
  core=bounded(['t','transform','derivative','absolute_value_terms','absolute_derivative_terms'])+['arithmetic_precision_bits']
  measured={'anyOf':[{'properties':{'values':{'required':['transform']}}},{'properties':{'outcome':enum('unresolved_derivative','channels_resolved_budget_unassessed','conditional_budget_met','conditional_budget_not_met','conditional_budget_unresolved')}}]}
  conditional={'properties':{'outcome':{'const':'conditional_budget_met'}}}
  extras=bounded(['conditional_root_distance_allowance','conditional_slope_margin','declared_radius','conditional_budget_target','declared_source_value_error','declared_source_derivative_error','conditional_value_numerator'])
  rows={'items':{'allOf':[{'if':measured,'then':{'properties':{'values':{'required':core}}}},{'if':conditional,'then':{'properties':{'values':{'required':extras}}}}]}}
  present={'properties':{'outcome':enum('point_measurement','partial_unresolved')}}
  data_fields=bounded(['finite_curvature_expression','relative_tolerance'])+['conditional_contiguous_prefix']
  schema['allOf'].append({'if':current,'then':{'properties':{'data':{'properties':{'rows':rows},'allOf':[{'if':present,'then':{'properties':{'values':{'required':data_fields}}}}]}}}})
 if diagnostic=='weighted_tail':
  request['properties']['atom_coordinate_serialization']={'const':'promoted_source_point_v1'};require_from(6,request_fields=['atom_coordinate_serialization'])
 if diagnostic=='energy_allowance':
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new='finite_stored_point_arithmetic_enclosures; conditional block expressions; external block bounds and hypotheses not certified'
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[11:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  core=['upper_trial_energy','low_block_lower_bound','high_block_lower_bound','cross_block_norm_bound','denominator']
  def bounded(names):return names+[name+'_'+side for name in names for side in ['lower','upper']]
  measured={'properties':{'semantics':enum(*RETAINED_REVISIONS[11:]),'data':{'properties':{'outcome':enum('conditional_bound_expression','sufficient_bound_unavailable')}}}}
  schema['allOf'].append({'if':measured,'then':{'properties':{'data':{'properties':{'values':{'required':bounded(core)+['arithmetic_precision_bits']}}}}}})
  measured={'properties':{'semantics':enum(*RETAINED_REVISIONS[11:]),'data':{'properties':{'outcome':{'const':'conditional_bound_expression'}}}}}
  schema['allOf'].append({'if':measured,'then':{'properties':{'data':{'properties':{'values':{'required':bounded(['conditional_energy_allowance','conditional_vector_allowance','trial_energy_magnitude'])}}}}}})
  scale_present={'properties':{'semantics':enum(*RETAINED_REVISIONS[11:]),'data':{'properties':{'values':{'required':['allowance_to_trial_energy_magnitude']}}}}}
  schema['allOf'].append({'if':scale_present,'then':{'properties':{'data':{'properties':{'values':{'required':bounded(['allowance_to_trial_energy_magnitude','allowance_below_trial_energy_magnitude','scale_comparison_resolved'])}}}}}})
 if diagnostic=='directional_response':
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new='finite_stored_point_arithmetic_enclosures; response ratios conditional on root, simple-minimum and displacement hypotheses; no source-error or ground-selection certificate'
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[10:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  core=['t','tau','directional_energy','direction_norm_squared','orthogonality_defect','rational_root_condition','root_condition_tolerance']
  fields=core+[name+'_'+side for name in core for side in ['lower','upper']]+['arithmetic_precision_bits']
  measured={'properties':{'outcome':enum('point_measurement','unresolved_denominator','channels_resolved_budget_unassessed')}}
  rows={'items':{'allOf':[{'if':measured,'then':{'properties':{'values':{'required':fields}}}}]}}
  schema['allOf'].append({'if':current,'then':{'properties':{'data':{'properties':{'rows':rows}}}}})
 if diagnostic=='arithmetic_energy':
  # Archived requests omit this identity; new requests bind the existing
  # explicit-operator precedence into their content-addressed cache key.
  request['properties']['component_selection']={'const':'explicit_operators_else_compact_actions_v1'}
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new='finite_stored_point_arithmetic_enclosures; excludes source-construction and operator-model errors; no ground-selection or convergence claim'
  data['properties']['assurance']=enum(old,new)
  current={'properties':{'semantics':enum(*RETAINED_REVISIONS[9:])},'required':['semantics']}
  schema.setdefault('allOf',[]).append({'if':current,'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  core=['total_tau_energy','sum_component_energy','sum_absolute_component_energy','energy_closure_defect','operator_action_closure_norm','retained_weil_energy']
  fields=core+[name+'_'+side for name in core for side in ['lower','upper']]+['arithmetic_precision_bits']
  measured={'properties':{'semantics':enum(*RETAINED_REVISIONS[9:]),'data':{'properties':{'outcome':enum('point_measurement','partial_unresolved')}}}}
  schema['allOf'].append({'if':measured,'then':{'properties':{'data':{'properties':{'values':{'required':fields},'rows':{'items':{'properties':{'values':{'required':['energy','energy_lower','energy_upper']}}}}}}}}})
 if diagnostic=='weighted_reference_projection':
  old='point_diagnostics_only; external_inputs_and_allowances_not_certified; no_RH_or_ground_selection_premise'
  new='finite_grid_arithmetic_enclosures; exact stored source/reference points; excludes source-construction and quadrature errors; no continuum or limit claim'
  data['properties']['assurance']=enum(old,new)
  schema.setdefault('allOf',[]).append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[6:])}},'then':{'properties':{'data':{'properties':{'assurance':{'const':new}}}}},'else':{'properties':{'data':{'properties':{'assurance':{'const':old}}}}}})
  core=['source_raw_center','weighted_l1','weighted_l2_squared','signed_integral']
  bound_fields=[name+'_'+side for name in core for side in ['lower','upper']]+['arithmetic_precision_bits']
  schema.setdefault('allOf',[]).append({'if':{'properties':{'semantics':enum(*RETAINED_REVISIONS[6:]),'data':{'properties':{'outcome':enum('point_measurement','rank_or_precision_unresolved')}}}},'then':{'properties':{'data':{'properties':{'values':{'required':core+bound_fields}}}}}})

def filename(kind):return kind.replace('_','-')+'-v1.schema.json'
def generate(output=None):
 output=Path(output) if output else ROOT/'docs/schemas'
 output.mkdir(parents=True,exist_ok=True)
 preparation={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Source-independent research reference preparation v1',**PREPARATION}
 (output/'ccm-reference-preparation-v1.schema.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf8',newline='\n')
 for kind,(_,_,data) in KINDS.items():
  schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':kind+' v1','description':'Source-bound observations; shape validation does not establish numerical correctness, ground selection, or convergence.',**obj(dict(schema_version={'const':1},semantics={'const':'ccm-retained-research-observations-v1'},kind={'const':kind},scope={'const':'finite_point_inputs; no_source_error_enclosure; no_ground_selection_or_convergence_certificate'},source_dependencies=array(DEP),request={'type':'object'},data=copy.deepcopy(data)))}
  if kind in BESPOKE_REQUESTS:schema['properties']['request']=BESPOKE_REQUESTS[kind]
  for diagnostic,extended_kind,_,_ in EXTENDED:
   if kind==extended_kind:measurement_contract(schema,diagnostic)
  revision_contract(schema,kind)
  if kind=='ccm_transform_enclosure':schema['properties']['scope']={'const':'finite_retained_function_enclosures; source_scope_explicit; no_infinite_limit_claim'}
  (output/filename(kind)).write_text(json.dumps(schema,indent=2)+'\n',encoding='utf8',newline='\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();generate(a.output)
