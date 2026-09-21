#!/usr/bin/env python3
"""Generate shared v1 retained-research payload schemas; never edits artifact indexes."""
import json, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S={'type':'string'}
SC={'type':'string','pattern':r'^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$'}
D={'type':'string','pattern':'^[0-9a-f]{64}$'}
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
 'ccm_operator_energy_analysis':('ccm-evidence',False,obj(dict(lambda_squared=SC,n_modes=I,precision_bits=I,source_eigenvalue=SC,coefficient_norm_squared=SC,rayleigh_quotient=SC,eigenvalue_defect=SC,relative_residual=SC,sum_absolute_energy_terms=SC,cancellation_digits=nullable(SC),component_decomposition=S,ground_selection=S))),
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
CERT={'type':'object','description':'PortableCcmSectorGapCertificate schema 3; the Toolkit replays exact-rational inertia and all identity checks before deriving any finite source-error allowance. This container schema alone does not validate its proof.','required':['schema_version','lambda_squared','n_modes','cutoff_free_tau','full_matrix_inertia_certificate','even_ground','even_first_excited','odd_ground','parity_invariance_premise','claim_scope'],'properties':{'schema_version':{'const':3},'lambda_squared':SC,'n_modes':I}}
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
  request['properties']['enclosure_algorithm']=enum('centered-taylor-48-integral-remainder-v1','centered-taylor-48-integral-remainder-v2')
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

def filename(kind):return kind.replace('_','-')+'-v1.schema.json'
def generate(output=None):
 output=Path(output) if output else ROOT/'docs/schemas'
 output.mkdir(parents=True,exist_ok=True)
 preparation={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Source-independent research reference preparation v1',**PREPARATION}
 (output/'ccm-reference-preparation-v1.schema.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf8',newline='\n')
 for kind,(_,_,data) in KINDS.items():
  schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':kind+' v1','description':'Source-bound observations; shape validation does not establish numerical correctness, ground selection, or convergence.',**obj(dict(schema_version={'const':1},semantics={'const':'ccm-retained-research-observations-v1'},kind={'const':kind},scope={'const':'finite_point_inputs; no_source_error_enclosure; no_ground_selection_or_convergence_certificate'},source_dependencies=array(DEP),request={'type':'object'},data=data))}
  if kind in BESPOKE_REQUESTS:schema['properties']['request']=BESPOKE_REQUESTS[kind]
  for diagnostic,extended_kind,_,_ in EXTENDED:
   if kind==extended_kind:measurement_contract(schema,diagnostic)
  if kind=='ccm_transform_enclosure':schema['properties']['scope']={'const':'finite_retained_function_enclosures; source_scope_explicit; no_infinite_limit_claim'}
  (output/filename(kind)).write_text(json.dumps(schema,indent=2)+'\n',encoding='utf8',newline='\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();generate(a.output)
