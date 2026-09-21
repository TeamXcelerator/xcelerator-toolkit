#!/usr/bin/env python3
"""Synthetic acceptance test for additive backfill. No scientific data or network."""
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary",type=Path);p.add_argument("--packets",type=Path,required=True)
    a=p.parse_args();a.binary=a.binary.resolve();a.packets.mkdir(parents=True,exist_ok=True)
    def enc(v):return json.dumps(v,separators=(",",":")).encode()
    def digest(v):return hashlib.sha256(v).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ccm-backfill-test-") as temp:
     root=Path(temp);approved=[]
     def write(name,kind,value,parents=()):
      data=enc(value);d=digest(data);approved.append(d)
      ver=dict(major=0,minor=15,patch=0,prerelease=None)
      m=dict(schema_version=1,key=dict(kind=kind,logical_key="synthetic-"+name,parameters_digest=digest(name.encode())),content_digest=d,size_bytes=len(data),objects=[dict(content_digest=d,size_bytes=len(data))],created_unix_seconds=1,producer_toolkit_version=ver,minimum_reader_version=ver,maximum_reader_version=None,quality="validated",visibility="local",immutable=True,dependencies=[dict(key=q["key"],content_digest=q["content_digest"],required_quality="validated") for q in parents],tags={},provenance_digest=None)
      (root/(name+".manifest.json")).write_bytes(enc(m));(root/(name+".payload.json")).write_bytes(data)
      return dict(manifest=name+".manifest.json",payload=name+".payload.json"),m
     tau,tm=write("tau","ccm_tau_matrix",dict(schema_version=2,lambda_squared="9",n_modes=1,precision_bits=128,entries=["3","0","0","0","3","0","0","0","3"]))
     def state(name,n,e,parents=()):
      return write(name,"ccm_weil_eigenpair",dict(schema_version=3,lambda_squared="9",n_modes=n,precision_bits=128,force_even=True,eigenvalue=e,eigenvector=["0"]*n+["1"]+["0"]*n),parents)
     st,sm=state("state",1,"3",[tm]);st2,_=state("state2",2,"3.0001")
     sec,sc=write("secular","ccm_secular_source",dict(schema_version=1,lambda_squared="9",n_modes=1,precision_bits=128,force_even=True,eigenpair_content_digest=sm["content_digest"],normalization="sum_xi_equals_sqrt_log_lambda_squared"),[sm])
     roots,rm=write("roots","ccm_root_discovery_window",dict(schema_version=5,lambda_squared="9",n_modes=1,precision_bits=128,force_even=True,first_root_index=1,discovery_mode="independent",reference_seeds_used=False,completeness="partial",outcomes=[dict(status="converged",details=dict(value="14")),dict(status="failed",details=dict(reason="synthetic missing row"))]),[sc])
     ref=dict(schema_version=1,definition="synthetic constant Fourier function",lambda_squared="9",precision_bits=128,coefficients=["0","1","0"],approximation_scope="finite point coefficients; no approximation enclosure")
     dataset=dict(schema_version=1,role="evaluation_points",attribution="synthetic test points",coordinate="mellin_t",precision_bits=128,points=[dict(ordinal=1,value="14",source_status="supplied")])
     obs=dict(schema_version=1,original_utf8="Synthetic reported result, not a numerical proof.",attribution="software fixture",definition="reported observation",hypotheses=[],borrowed_inputs=[],limitations=["not replayed"])
     tasks=[("geometry",dict(operation="state_geometry",state=st)),("energy",dict(operation="operator_energy",state=st,matrix=tau)),("roots",dict(operation="root_window",state=st,roots=roots,secular=sec)),("transform",dict(operation="indexed_transform",state=st,roots=roots,secular=sec)),("reference",dict(operation="reference_source",spec=ref)),("dataset",dict(operation="reference_dataset",spec=dataset)),("projection",dict(operation="reference_projection",state=st,inputs=dict(schema_version=1,reference=ref,basis=[ref],projection=dict(working_precision_bits=192,normalization="unit_l2_dx",fixed_second_component=None)))),("observation",dict(operation="observation_packet",observation=obs)),("stabilization",dict(operation="stabilization",states=[st,st2],options=dict(working_precision_bits=192,relative_tolerance="0.01",consecutive_steps=1)))]
     extended=dict(schema_version=1,source_eigenpair=sm["content_digest"],lambda_squared="9",n_modes=1,precision_bits=128,convention_id="synthetic fixture",definition_digest=digest(b"synthetic"),approximation_scope="finite fixture data",
      target=dict(definition_digest=digest(b"sampled fixture"),evaluation_policy="uniform x grid",approximation_scope="point samples",intervals=8,values=["1"]*9,basis_values=[["1"]*9],fixed_second_component=None,raw_normalizer="1",trial_coefficients=["0","1","0"]),
      reference_jets=[dict(ordinal=1,t="14",reference_window=dict(value="0",derivative="0"),reference_full=dict(value="0",derivative="0"),exterior_tail=dict(value="0",derivative="0"),endpoint_tail_part=None,fitted_interior_parts=[],error_normalization=None,source_value_error=None,tail_value_error=None,source_derivative_error=None,root_separation_radius=None)],
      components=[dict(label="diagonal",source_digest=digest(b"diagonal"),diagonal=["3"]*3,dense=[],rank_one=[])],components_are_complete=True,perturbations=[dict(label="identity",source_digest=digest(b"identity"),diagonal=["1"]*3,dense=[],rank_one=[])],deficit="0.5",deficit_kind="fuchs_approximation",
      atoms=[dict(ordinal=1,coordinate="1",weight="2",family="zero",partition="declared_band"),dict(ordinal=1,coordinate="2",weight="-1",family="lattice",partition="declared_band")],atom_coordinate="positive fixture coordinate",atom_coverage="finite supplied atoms",tail_checkpoints=["1","2"],
      cluster=[dict(source_digest=sm["content_digest"],n_modes=1,precision_bits=128,eigenvalue="3",coefficients=["0","1","0"],assembly_policy="fixture")],previous_cluster=[],cluster_boundary_eigenvalues=["3","4"],
      energy_allowance=dict(upper_trial_energy="3",low_block_lower_bound="2",high_block_lower_bound="5",cross_block_norm_bound="1",hypothesis_record_digest=digest(b"conditional fixture"),hypotheses=["supplied point bounds"]))
     extended['run_once']=dict(tail_form=dict(definition_digest=digest(b"synthetic tail"),dimension=1,finite_zero_form=["1"],tail_correction=["0.5"],lattice_gram=["1"],tail_operator_error=None,coverage="synthetic finite model",hypotheses=[]))
     chunk_bytes=b''.join(enc(dict(coordinate=str(x),signed_weight="1",family="zero"))+b"\n" for x in [1,2,3,4])
     (root/"band-atoms.jsonl").write_bytes(chunk_bytes)
     extended['run_once']['completion']=dict(band=dict(degree=2,coordinate="positive fixture coordinate",definition_digest=digest(b"finite positive band"),atoms=[],coverage="finite four-atom grid",hypotheses=["finite positive fixture"],borrowed_inputs=[],input_energy=None,scoring_roots=[]),atom_analysis=dict(maximum_atoms=300000,maximum_input_bytes=len(chunk_bytes),band_chunks=[dict(relative_path="band-atoms.jsonl",sha256=digest(chunk_bytes),bytes=len(chunk_bytes),rows=4)],cutoffs=["1","3"],evaluations=[dict(ordinal=1,coordinate="1",label="declared fixture point",exclude=[dict(ordinal=1,family="zero",partition="declared_band")])]))
     external_bytes=enc(extended);external_path=root/"external.json";external_path.write_bytes(external_bytes)
     preparation=dict(schema_version=1,lambda_squared="9",precision_bits=128,definition_digest=digest(b"finite constant preparation"),approximation_scope="synthetic finite-only reference",finite_reference=dict(schema_version=1,reference=ref,basis=[],projection=dict(working_precision_bits=192,normalization="center_one",fixed_second_component=None)))
     preparation_bytes=enc(preparation);(root/"preparation.json").write_bytes(preparation_bytes)
     for diagnostic in ["external_source","compactness","weighted_reference_projection","signed_transform","arithmetic_energy","directional_response","weighted_tail","spectral_cluster","resolution_budget","energy_allowance","complex_transform","root_transport","operator_cluster","finite_section_transfer","tail_operator","observable_budget","capture_preflight","consistency","configuration_comparison","band_reconstruction","transform_enclosure"]:
      tasks.append((diagnostic,dict(operation="extended_research",diagnostic=diagnostic,state=st,matrix=tau if diagnostic in ["arithmetic_energy","directional_response","root_transport","operator_cluster","finite_section_transfer"] else None,roots=roots if diagnostic in ["directional_response","resolution_budget","complex_transform","root_transport","observable_budget"] else None,secular=sec if diagnostic in ["directional_response","resolution_budget","complex_transform","root_transport","observable_budget"] else None,parent_manifests=[],input="preparation.json" if diagnostic=="weighted_reference_projection" else "external.json",input_sha256=digest(preparation_bytes if diagnostic=="weighted_reference_projection" else external_bytes),options=None)))
     batch=dict(schema_version=1,approved_payload_digests=approved,cache_root="cache",jobs=[dict(id=i,task=t) for i,t in tasks])
     plan=root/"batch.json";plan.write_bytes(enc(batch));out=root/"output"
     def run(plan=plan,out=out):return subprocess.run([str(a.binary),str(plan),str(out)],capture_output=True,text=True)
     cold=run();assert cold.returncode==0,cold.stderr
     saved={i:(out/(i+".json")).read_bytes() for i,_ in tasks}
     warm=run();assert warm.returncode==0,warm.stderr
     assert all((out/(i+".json")).read_bytes()==b for i,b in saved.items())
     for i,b in saved.items():(a.packets/(i+".json")).write_bytes(b)
     summary=json.loads(sorted(out.glob("summary-*.json"))[-1].read_text())
     assert next(x for x in summary["outcomes"] if x["id"]=="transform")["row_outcomes"]["missing_input"]==1
     assert any(row["label"]=="band_cutoff" for row in json.loads(saved['band_reconstruction'])["report"]["data"]["rows"])
     assert any(row["label"]=="signed_atom_kernel" for row in json.loads(saved['weighted_tail'])["report"]["data"]["rows"])
     (root/"band-atoms.jsonl").write_bytes(chunk_bytes.replace(b'"1"',b'"9"',1))
     assert run().returncode!=0
     assert all((out/(i+".json")).read_bytes()==b for i,b in saved.items())
     (root/"band-atoms.jsonl").write_bytes(chunk_bytes)
     assert run().returncode==0
     # Changed external input bytes cannot alter a frozen historical batch.
     external_path.write_bytes(external_bytes+b" ")
     assert run().returncode!=0
     assert all((out/(i+".json")).read_bytes()==b for i,b in saved.items())
     external_path.write_bytes(external_bytes)
     assert run().returncode==0
     # A bad source cannot erase successful independent jobs or existing children.
     source=root/st["payload"];original=source.read_bytes();source.write_bytes(original+b" ")
     bad=run();assert bad.returncode!=0 and "INCOMPLETE" in bad.stderr
     assert all((out/(i+".json")).read_bytes()==b for i,b in saved.items())
     failed=json.loads(sorted(out.glob("summary-*.json"))[-1].read_text())
     assert failed["failed_jobs"]>0 and any(x.get("status")=="retained" for x in failed["outcomes"])
     source.write_bytes(original)
     assert run().returncode==0
     # A different batch cannot reuse the frozen output directory.
     altered=dict(batch);altered["jobs"]=batch["jobs"][:-1]
     other=root/"other.json";other.write_bytes(enc(altered));assert run(other).returncode!=0
     # A modified result is never overwritten, even when the managed cache is sound.
     target=out/"energy.json";changed=json.loads(target.read_text());changed["report"]["data"]["rayleigh_quotient"]="999";target.write_bytes(enc(changed));before=target.read_bytes()
     assert run().returncode!=0 and target.read_bytes()==before
     result=dict(scope="synthetic offline backfill acceptance; not historical campaign repair",managed_kinds=30,source_independent_preparation=True,external_input_digest_enforced=True,cold_warm_byte_identity=True,failed_rows_retained=True,independent_jobs_survive_failure=True,corrupt_source_rejected=True,frozen_batch_enforced=True,modified_output_preserved=True,chunk_hash_enforced=True,cutoff_and_signed_atom_rows_retained=True,attempts_append_only=len(list(out.glob("summary-*.json")))==9)
     assert result["attempts_append_only"]
     (a.packets/"acceptance.json").write_text(json.dumps(result,indent=2)+"\n")
     print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
