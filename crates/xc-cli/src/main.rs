use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use xc_cache::{
    abandon_publication_target, advance_publication_target, audit_remote_shard,
    coordinate_discovered_publication, discover_remote_publication_routing, execute_local_prune,
    execute_revocation_update, execute_shard_index_repair, execute_supersession_update,
    execute_topology_rollover, export_cache_bundle, inspect_publication_recovery,
    materialize_cache_bundle_artifact, materialize_resolved_remote_artifact,
    materialize_resolved_remote_artifact_closure, package_canonical_payload_zip64,
    plan_cache_derivations, plan_deduplication, plan_local_prune, plan_revocation_update,
    plan_shard_index_repair, plan_storage_placement, plan_supersession_update,
    plan_topology_rollover, reconstruct_transport_package, record_remote_cache_access,
    resolve_remote_semantic_artifact, validate_remote_publication_routes, verify_cache_bundle,
    verify_canonical_payload_zip64, verify_live_github_acceptance, ArtifactAssuranceState,
    ArtifactCopyEvidence, CacheBundleArtifactIdentity, CacheBundleConsumptionPolicy,
    CacheBundleExportRequest, CacheBundleExportSource, CacheBundlePolicy, CacheError,
    CacheNetworkRegistry, CachePlanRequest, CachePublicationPolicy, CacheVisibility,
    CanonicalPayloadEnvelope, CapacityLedger, DeduplicationPlanningRequest, DurabilityPolicy,
    GitCliRemoteStore, GitHubCredentialApiProbe, GitHubRepositoryEndpoint,
    LiveGitHubAcceptanceArtifact, LocalPruneCandidate, LocalPrunePolicy, PayloadFileSource,
    ProjectedPublicationAddition, PublicationDestination, PublicationFinalizationPolicy,
    PublicationJournalStore, PublicationMetadataBundle, PublicationTargetExecution,
    PublicationTargetState, RemoteArtifactClosureMaterializationReport,
    RemoteArtifactMaterializationReport, RemoteCacheAccessProvenanceRequest,
    RemoteFabricTrustPolicy, RemoteGitStore, RemoteResolverOverlay, RemoteSemanticQuery,
    RemoteShardAuditReport, RemoteShardReader, RemoteTargetPublicationPlanningInput,
    RemoteTopologySource, RevocationRecord, RevocationUpdateOutcome, RevocationUpdatePlan,
    SemanticResolutionReport, ShardAuditPolicy, ShardIndexRepairOutcome, ShardIndexRepairPlan,
    ShardIndexRepairPolicy, StoragePlacementRequest, SuccessorShardReadinessEvidence,
    SupersessionRecord, SupersessionUpdateOutcome, SupersessionUpdatePlan, TopologyRolloverOutcome,
    TopologyRolloverPlan, TopologyTrustPolicy, TransportEncodingRecord, TransportPolicy,
};
use xc_core::{
    CacheAccessProvenance, CacheReuseDisposition, CacheValidationMode as ProvenanceValidationMode,
    CacheValidationOutcome, CancellationToken, FailureDiagnostic, PublicationTarget,
    ResourcePolicy, RetryClassification,
};

const COMMAND_DOCUMENT_MAX_BYTES: u64 = 16 * 1024 * 1024;

fn main() {
    match run() {
        Ok(()) => {}
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&error.report())
                    .unwrap_or_else(|_| "{\"status\":\"error\"}".to_owned())
            );
            std::process::exit(error.exit_code());
        }
    }
}

fn run() -> Result<(), CliError> {
    let arguments = std::env::args_os()
        .skip(1)
        .map(|argument| {
            argument.into_string().map_err(|_| {
                CliError::usage("command arguments must be valid Unicode on this platform")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    match parse_command(&arguments)? {
        Command::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Command::AuthProbe { repository } => {
            let session = GitHubCredentialApiProbe::default().probe_repository(&repository)?;
            write_success("cache.auth-probe", session.evidence())
        }
        Command::Package { request } => {
            let request: PackageRequest = load_document(&request)?;
            request.validate()?;
            let report = package_canonical_payload_zip64(
                &request.envelope,
                &request.sources,
                &request.destination,
                &request.resources,
                &CancellationToken::for_policy(&request.resources),
            )?;
            write_success("cache.package", &report)
        }
        Command::Reconstruct { request } => {
            let request: ReconstructRequest = load_document(&request)?;
            request.validate()?;
            let report = reconstruct_transport_package(
                &request.encoding,
                &request.parts_root,
                &request.destination,
                &request.resources,
                &CancellationToken::for_policy(&request.resources),
            )?;
            write_success("cache.reconstruct", &report)
        }
        Command::VerifyPackage { request } => {
            let request: VerifyPackageRequest = load_document(&request)?;
            request.validate()?;
            let report = verify_canonical_payload_zip64(
                &request.envelope,
                &request.encoding,
                &request.package_path,
                &CancellationToken::for_policy(&request.resources),
            )?;
            write_success("cache.verify-package", &report)
        }
        Command::Transaction {
            journal_root,
            transaction_id,
        } => {
            let journal =
                PublicationJournalStore::new(journal_root).load_latest(&transaction_id)?;
            write_success("cache.transaction", &journal)
        }
        Command::Resume { request } => resume_publication(&load_document(&request)?),
        Command::Abandon { request } => abandon_publication(&load_document(&request)?),
        Command::Publish { request } => publish_artifact(&load_document(&request)?),
        Command::Find { request } => find_artifact(&load_document(&request)?),
        Command::Inspect { request } => inspect_artifact(&load_document(&request)?),
        Command::Validate { request } => validate_artifact(&load_document(&request)?),
        Command::ShardStatus { request } => shard_status(&load_document(&request)?),
        Command::Revoke { request } => revoke_identity(&load_document(&request)?),
        Command::Supersede { request } => supersede_artifact(&load_document(&request)?),
        Command::Rollover { request } => rollover_shard(&load_document(&request)?),
        Command::Fetch { request } => fetch_artifact(&load_document(&request)?),
        Command::Audit { request } => audit_shard(&load_document(&request)?),
        Command::RepairIndex { request } => repair_shard_index(&load_document(&request)?),
        Command::Prune { request } => plan_prune(&load_document(&request)?),
        Command::Plan { request } => plan_cache(&load_document(&request)?),
        Command::PlacementPlan { request } => plan_placement(&load_document(&request)?),
        Command::DedupPlan { request } => plan_dedup(&load_document(&request)?),
        Command::Export { request } => export_bundle(&load_document(&request)?),
        Command::VerifyBundle { request } => verify_bundle(&load_document(&request)?),
        Command::VerifyLiveAcceptance { request } => {
            verify_live_acceptance(&load_document(&request)?)
        }
        Command::ConsumeBundle { request } => consume_bundle(&load_document(&request)?),
    }
}

const USAGE: &str = r#"Xcelerator Toolkit v0.13.4

Usage:
  xc cache auth-probe OWNER/REPOSITORY
  xc cache package REQUEST.json
  xc cache reconstruct REQUEST.json
  xc cache verify-package REQUEST.json
  xc cache transaction JOURNAL_ROOT TRANSACTION_ID
  xc cache find REQUEST.json
  xc cache inspect REQUEST.json
  xc cache validate REQUEST.json
  xc cache shard-status REQUEST.json
  xc cache revoke REQUEST.json
  xc cache supersede REQUEST.json
  xc cache rollover REQUEST.json
  xc cache fetch REQUEST.json
  xc cache audit REQUEST.json
  xc cache repair-index REQUEST.json
  xc cache prune REQUEST.json
  xc cache plan REQUEST.json
  xc cache placement-plan REQUEST.json
  xc cache dedup-plan REQUEST.json
  xc cache export REQUEST.json
  xc cache verify-bundle REQUEST.json
  xc cache verify-live-acceptance RECORD.json
  xc cache consume-bundle REQUEST.json
  xc cache publish REQUEST.json
  xc cache resume REQUEST.json
  xc cache abandon REQUEST.json

There is no implicit remote publication. A resume request is read-only unless
it contains execute_remote_mutations=true and a complete execution block.
Index repair is read-only unless execute_remote_mutations=true and
confirm_repair=true; stale plans stop on ref conflict without automatic retry.
Abandonment requires confirm_abandon=true, fresh write permission, and proof
that no receipt is visible at the current remote head."#;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Command {
    Help,
    AuthProbe {
        repository: String,
    },
    Package {
        request: PathBuf,
    },
    Reconstruct {
        request: PathBuf,
    },
    VerifyPackage {
        request: PathBuf,
    },
    Transaction {
        journal_root: PathBuf,
        transaction_id: String,
    },
    Resume {
        request: PathBuf,
    },
    Publish {
        request: PathBuf,
    },
    Find {
        request: PathBuf,
    },
    Inspect {
        request: PathBuf,
    },
    Validate {
        request: PathBuf,
    },
    ShardStatus {
        request: PathBuf,
    },
    Revoke {
        request: PathBuf,
    },
    Supersede {
        request: PathBuf,
    },
    Rollover {
        request: PathBuf,
    },
    Fetch {
        request: PathBuf,
    },
    Audit {
        request: PathBuf,
    },
    RepairIndex {
        request: PathBuf,
    },
    Prune {
        request: PathBuf,
    },
    Plan {
        request: PathBuf,
    },
    PlacementPlan {
        request: PathBuf,
    },
    DedupPlan {
        request: PathBuf,
    },
    Export {
        request: PathBuf,
    },
    VerifyBundle {
        request: PathBuf,
    },
    VerifyLiveAcceptance {
        request: PathBuf,
    },
    ConsumeBundle {
        request: PathBuf,
    },
    Abandon {
        request: PathBuf,
    },
}

fn parse_command(arguments: &[String]) -> Result<Command, CliError> {
    match arguments {
        [] => Ok(Command::Help),
        [help] if help == "help" || help == "--help" || help == "-h" => Ok(Command::Help),
        [cache, help] if cache == "cache" && (help == "help" || help == "--help") => {
            Ok(Command::Help)
        }
        [cache, command, repository] if cache == "cache" && command == "auth-probe" => {
            Ok(Command::AuthProbe {
                repository: repository.clone(),
            })
        }
        [cache, command, request] if cache == "cache" && command == "package" => {
            Ok(Command::Package {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "reconstruct" => {
            Ok(Command::Reconstruct {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "verify-package" => {
            Ok(Command::VerifyPackage {
                request: PathBuf::from(request),
            })
        }
        [cache, command, root, transaction_id] if cache == "cache" && command == "transaction" => {
            Ok(Command::Transaction {
                journal_root: PathBuf::from(root),
                transaction_id: transaction_id.clone(),
            })
        }
        [cache, command, request] if cache == "cache" && command == "resume" => {
            Ok(Command::Resume {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "publish" => {
            Ok(Command::Publish {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "find" => Ok(Command::Find {
            request: PathBuf::from(request),
        }),
        [cache, command, request] if cache == "cache" && command == "inspect" => {
            Ok(Command::Inspect {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "validate" => {
            Ok(Command::Validate {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "shard-status" => {
            Ok(Command::ShardStatus {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "revoke" => {
            Ok(Command::Revoke {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "supersede" => {
            Ok(Command::Supersede {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "rollover" => {
            Ok(Command::Rollover {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "fetch" => Ok(Command::Fetch {
            request: PathBuf::from(request),
        }),
        [cache, command, request] if cache == "cache" && command == "audit" => Ok(Command::Audit {
            request: PathBuf::from(request),
        }),
        [cache, command, request] if cache == "cache" && command == "repair-index" => {
            Ok(Command::RepairIndex {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "prune" => Ok(Command::Prune {
            request: PathBuf::from(request),
        }),
        [cache, command, request] if cache == "cache" && command == "plan" => Ok(Command::Plan {
            request: PathBuf::from(request),
        }),
        [cache, command, request] if cache == "cache" && command == "placement-plan" => {
            Ok(Command::PlacementPlan {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "dedup-plan" => {
            Ok(Command::DedupPlan {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "export" => {
            Ok(Command::Export {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "verify-bundle" => {
            Ok(Command::VerifyBundle {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "verify-live-acceptance" => {
            Ok(Command::VerifyLiveAcceptance {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "consume-bundle" => {
            Ok(Command::ConsumeBundle {
                request: PathBuf::from(request),
            })
        }
        [cache, command, request] if cache == "cache" && command == "abandon" => {
            Ok(Command::Abandon {
                request: PathBuf::from(request),
            })
        }
        _ => Err(CliError::usage("unknown or incomplete command")),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationResumeRequest {
    schema_version: u32,
    journal_root: PathBuf,
    transaction_id: String,
    execute_remote_mutations: bool,
    transport: Option<GitTransportRequest>,
    route_guard: Option<PublicationRouteGuardRequest>,
    execution: Option<PublicationExecutionRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationExecutionRequest {
    staging_root: PathBuf,
    resources: ResourcePolicy,
    maximum_steps: u32,
    targets: BTreeMap<PublicationDestination, PublicationTargetExecutionRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationTargetExecutionRequest {
    finalization_policy: PublicationFinalizationPolicy,
    metadata_bundle: PublicationMetadataBundle,
    maximum_index_bytes: u64,
    receipt_verified_at_unix_seconds: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GitTransportRequest {
    temporary_root: PathBuf,
    staged_parts_root: PathBuf,
    author_name: String,
    author_email: String,
    resources: ResourcePolicy,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GitReadTransportRequest {
    temporary_root: PathBuf,
    resources: ResourcePolicy,
}

impl GitReadTransportRequest {
    fn open(&self) -> Result<GitCliRemoteStore, CliError> {
        if self.temporary_root.as_os_str().is_empty() {
            return Err(CliError::input(
                "read-only Git transport temporary_root is required",
            ));
        }
        Ok(GitCliRemoteStore::new(
            &self.temporary_root,
            self.temporary_root.join("read-only-staging"),
            "Xcelerator read-only resolver",
            "resolver@invalid",
        )?
        .with_resource_policy(self.resources.clone()))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationRouteGuardRequest {
    family: String,
    publication_policy: CachePublicationPolicy,
    topology_source: RemoteTopologySource,
    topology_trust: TopologyTrustPolicy,
    fabric_trust: RemoteFabricTrustPolicy,
    evaluation_unix_seconds: u64,
    network: CacheNetworkRegistry,
}

struct PublicationRouteGuard<'a> {
    family: &'a str,
    publication_policy: &'a CachePublicationPolicy,
    topology_source: &'a RemoteTopologySource,
    topology_trust: &'a TopologyTrustPolicy,
    fabric_trust: &'a RemoteFabricTrustPolicy,
    evaluation_unix_seconds: u64,
    network: &'a CacheNetworkRegistry,
}

impl PublicationRouteGuardRequest {
    fn validate(&self) -> Result<(), CliError> {
        if self.family.trim().is_empty() {
            return Err(CliError::input(
                "publication route guard family is required",
            ));
        }
        self.publication_policy.digest()?;
        self.topology_source.validate()?;
        self.network.validate()?;
        Ok(())
    }
}

impl GitTransportRequest {
    fn validate(&self) -> Result<(), CliError> {
        if self.temporary_root.as_os_str().is_empty()
            || self.staged_parts_root.as_os_str().is_empty()
            || self.author_name.trim().is_empty()
            || self.author_email.trim().is_empty()
        {
            return Err(CliError::input(
                "Git transport roots, author name, and author email are required",
            ));
        }
        Ok(())
    }

    fn open(&self) -> Result<GitCliRemoteStore, CliError> {
        self.validate()?;
        Ok(GitCliRemoteStore::new(
            &self.temporary_root,
            &self.staged_parts_root,
            &self.author_name,
            &self.author_email,
        )?
        .with_resource_policy(self.resources.clone()))
    }
}

impl PublicationResumeRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        validate_transaction_locator(&self.journal_root, &self.transaction_id)?;
        match (
            self.execute_remote_mutations,
            self.transport.as_ref(),
            self.route_guard.as_ref(),
            self.execution.as_ref(),
        ) {
            (false, None, None, None) => Ok(()),
            (true, Some(transport), Some(route_guard), Some(execution)) => {
                transport.validate()?;
                route_guard.validate()?;
                execution.validate()?;
                execution.validate_transport_binding(transport)
            }
            (false, _, _, _) => Err(CliError::input(
                "a read-only resume request must omit transport, route_guard, and execution blocks",
            )),
            (true, _, _, _) => Err(CliError::input(
                "execute_remote_mutations=true requires complete transport, route_guard, and execution blocks",
            )),
        }
    }
}

impl PublicationExecutionRequest {
    fn validate(&self) -> Result<(), CliError> {
        if self.staging_root.as_os_str().is_empty()
            || self.maximum_steps == 0
            || self.maximum_steps > 1_000_000
            || self.targets.is_empty()
        {
            return Err(CliError::input(
                "publication execution requires a staging root, targets, and 1..=1000000 maximum_steps",
            ));
        }
        for (destination, target) in &self.targets {
            target.finalization_policy.validate()?;
            if target.maximum_index_bytes == 0 || target.receipt_verified_at_unix_seconds == 0 {
                return Err(CliError::input(format!(
                    "publication execution target {destination:?} has invalid index or receipt-time settings"
                )));
            }
        }
        Ok(())
    }

    fn validate_transport_binding(&self, transport: &GitTransportRequest) -> Result<(), CliError> {
        if self.staging_root != transport.staged_parts_root || self.resources != transport.resources
        {
            return Err(CliError::input(
                "publication execution staging_root and resources must exactly match the Git transport staged_parts_root and resources",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationPublishRequest {
    schema_version: u32,
    family: String,
    target: PublicationTarget,
    journal_root: PathBuf,
    publication_policy: CachePublicationPolicy,
    topology_source: RemoteTopologySource,
    topology_trust: TopologyTrustPolicy,
    fabric_trust: RemoteFabricTrustPolicy,
    evaluation_unix_seconds: u64,
    network: CacheNetworkRegistry,
    encoding: TransportEncodingRecord,
    transport_policy: TransportPolicy,
    target_inputs: BTreeMap<PublicationDestination, RemoteTargetPublicationPlanningInput>,
    transport: GitTransportRequest,
    execute_remote_mutations: bool,
    execution: Option<PublicationExecutionRequest>,
}

impl PublicationPublishRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        if self.family.trim().is_empty() || self.journal_root.as_os_str().is_empty() {
            return Err(CliError::input(
                "publication family and journal_root are required",
            ));
        }
        self.publication_policy.digest()?;
        self.topology_source.validate()?;
        self.network.validate()?;
        self.encoding.validate()?;
        self.transport_policy.validate()?;
        self.transport.validate()?;
        let destinations = publication_destinations(self.target)?;
        if self.target_inputs.keys().copied().collect::<Vec<_>>() != destinations {
            return Err(CliError::input(
                "publication target_inputs must exactly match the requested targets",
            ));
        }
        match (self.execute_remote_mutations, self.execution.as_ref()) {
            (false, None) => Ok(()),
            (true, Some(execution)) => {
                execution.validate()?;
                execution.validate_transport_binding(&self.transport)?;
                if execution.targets.keys().copied().collect::<Vec<_>>() != destinations {
                    return Err(CliError::input(
                        "publication execution targets must exactly match the requested targets",
                    ));
                }
                for destination in &destinations {
                    let input = &self.target_inputs[destination];
                    let bundle = &execution.targets[destination].metadata_bundle;
                    if bundle.family != self.family
                        || bundle.encoding != self.encoding
                        || bundle.manifest.digest()? != input.candidate.manifest_digest
                        || bundle.manifest.semantic_digest != input.candidate.semantic_digest
                        || bundle.manifest.payload_digest != input.candidate.payload_digest
                        || bundle.validator_evidence != input.candidate.validator_evidence
                        || bundle.achieved_assurance != input.candidate.achieved_assurance
                        || bundle.disposition != input.candidate.disposition
                        || bundle.target_metadata != input.candidate.public_metadata
                    {
                        return Err(CliError::input(format!(
                            "publication execution bundle {destination:?} does not match its authorized candidate and transport"
                        )));
                    }
                }
                Ok(())
            }
            (false, Some(_)) => Err(CliError::input(
                "a dry-run publish request must omit the execution block",
            )),
            (true, None) => Err(CliError::input(
                "execute_remote_mutations=true requires a complete execution block",
            )),
        }
    }
}

#[derive(Serialize)]
struct PublicationPublishCommandReport {
    remote_mutation_enabled: bool,
    routing: xc_cache::RemotePublicationRoutingPlan,
    coordinated_plan: xc_cache::CoordinatedPublicationPlan,
    initial_checkpoint_path: Option<PathBuf>,
    execution: Option<PublicationResumeCommandReport>,
}

fn publish_artifact(request: &PublicationPublishRequest) -> Result<(), CliError> {
    request.validate()?;
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let additions = request
        .target_inputs
        .iter()
        .map(|(destination, input)| {
            (
                *destination,
                ProjectedPublicationAddition {
                    unique_payload_bytes: request.encoding.package_size_bytes,
                    metadata_bytes: input.projected_metadata_bytes,
                    projected_history_bytes: input.projected_history_bytes,
                },
            )
        })
        .collect();
    let routing = discover_remote_publication_routing(
        &remote,
        &cancellation,
        &request.family,
        request.target,
        &request.topology_source,
        &request.topology_trust,
        &request.fabric_trust,
        request.evaluation_unix_seconds,
        &request.network,
        &additions,
    )?;
    let probe = GitHubCredentialApiProbe::default();
    let mut sessions = BTreeMap::new();
    for (destination, selection) in &routing.selections {
        sessions.insert(
            *destination,
            probe.probe_repository(&selection.authorized_repository)?,
        );
    }
    let coordinated_plan = coordinate_discovered_publication(
        &routing,
        &request.publication_policy,
        &request.topology_trust,
        &request.network,
        &request.encoding,
        &request.transport_policy,
        &request.target_inputs,
        &sessions,
    )?;
    if !request.execute_remote_mutations {
        return write_success(
            "cache.publish",
            &PublicationPublishCommandReport {
                remote_mutation_enabled: false,
                routing,
                coordinated_plan,
                initial_checkpoint_path: None,
                execution: None,
            },
        );
    }

    let planned_journal = coordinated_plan.journal.as_ref().ok_or_else(|| {
        let reasons = coordinated_plan
            .target_reports
            .iter()
            .filter(|(_, report)| !report.accepted())
            .map(|(destination, report)| format!("{destination:?}: {}", report.reasons.join("; ")))
            .collect::<Vec<_>>()
            .join(" | ");
        CliError::input(format!(
            "publication preflight was not authorized: {reasons}"
        ))
    })?;
    let checkpoints = PublicationJournalStore::new(&request.journal_root);
    let (mut journal, initial_checkpoint_path) =
        match checkpoints.load_if_exists(&planned_journal.transaction_id)? {
            Some(existing) => (existing, None),
            None => {
                let path = checkpoints.save(planned_journal)?;
                (planned_journal.clone(), Some(path))
            }
        };
    let execution_request = request
        .execution
        .as_ref()
        .expect("validated mutating publish has execution settings");
    let execution = execute_publication_journal(
        &remote,
        &checkpoints,
        &mut journal,
        execution_request,
        &sessions,
        &PublicationRouteGuard {
            family: &request.family,
            publication_policy: &request.publication_policy,
            topology_source: &request.topology_source,
            topology_trust: &request.topology_trust,
            fabric_trust: &request.fabric_trust,
            evaluation_unix_seconds: request.evaluation_unix_seconds,
            network: &request.network,
        },
    )?;
    write_success(
        "cache.publish",
        &PublicationPublishCommandReport {
            remote_mutation_enabled: true,
            routing,
            coordinated_plan,
            initial_checkpoint_path,
            execution: Some(execution),
        },
    )
}

fn publication_destinations(
    target: PublicationTarget,
) -> Result<Vec<PublicationDestination>, CliError> {
    match target {
        PublicationTarget::None => Err(CliError::input(
            "cache publication target must be private, public, or both",
        )),
        PublicationTarget::Private => Ok(vec![PublicationDestination::Private]),
        PublicationTarget::Public => Ok(vec![PublicationDestination::Public]),
        PublicationTarget::Both => Ok(vec![
            PublicationDestination::Private,
            PublicationDestination::Public,
        ]),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheFindRequest {
    schema_version: u32,
    query: RemoteSemanticQuery,
    overlays: Vec<RemoteResolverOverlay>,
    transport: GitReadTransportRequest,
}

#[derive(Debug, Serialize)]
struct CacheLookupCommandReport {
    resolution: SemanticResolutionReport,
    cache_provenance: CacheAccessProvenance,
}

fn find_artifact(request: &CacheFindRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    request.query.validate()?;
    if request.overlays.is_empty() {
        return Err(CliError::input(
            "cache find requires at least one remote overlay",
        ));
    }
    let remote = request.transport.open()?;
    let report = resolve_remote_semantic_artifact(
        &remote,
        &CancellationToken::for_policy(&request.transport.resources),
        &request.query,
        &request.overlays,
    )?;
    let provenance = record_remote_cache_access(RemoteCacheAccessProvenanceRequest {
        operation: "cache.find",
        family: &request.query.family,
        overlays: &request.overlays,
        resolution: &report,
        reuse_disposition: CacheReuseDisposition::InspectedOnly,
        validation_mode: ProvenanceValidationMode::Fast,
        validation_outcome: if report.selected.is_some() {
            CacheValidationOutcome::Passed
        } else {
            CacheValidationOutcome::Failed
        },
        validation_detail: None,
        materialization: None,
    })?;
    write_success(
        "cache.find",
        &CacheLookupCommandReport {
            resolution: report,
            cache_provenance: provenance,
        },
    )
}

fn inspect_artifact(request: &CacheFindRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    request.query.validate()?;
    if request.overlays.is_empty() {
        return Err(CliError::input(
            "cache inspect requires at least one ordered remote overlay",
        ));
    }
    let remote = request.transport.open()?;
    let report = resolve_remote_semantic_artifact(
        &remote,
        &CancellationToken::for_policy(&request.transport.resources),
        &request.query,
        &request.overlays,
    )?;
    let provenance = record_remote_cache_access(RemoteCacheAccessProvenanceRequest {
        operation: "cache.inspect",
        family: &request.query.family,
        overlays: &request.overlays,
        resolution: &report,
        reuse_disposition: CacheReuseDisposition::InspectedOnly,
        validation_mode: ProvenanceValidationMode::Fast,
        validation_outcome: if report.selected.is_some() {
            CacheValidationOutcome::Passed
        } else {
            CacheValidationOutcome::Failed
        },
        validation_detail: None,
        materialization: None,
    })?;
    write_success(
        "cache.inspect",
        &CacheLookupCommandReport {
            resolution: report,
            cache_provenance: provenance,
        },
    )
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum CacheValidationMode {
    Fast,
    Full,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheValidateRequest {
    schema_version: u32,
    mode: CacheValidationMode,
    query: RemoteSemanticQuery,
    overlays: Vec<RemoteResolverOverlay>,
    transport: GitReadTransportRequest,
    parts_root: Option<PathBuf>,
    package_destination: Option<PathBuf>,
    dependency_packages_root: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct CacheValidateCommandReport {
    mode: CacheValidationMode,
    resolution: SemanticResolutionReport,
    materialization: Option<RemoteArtifactClosureMaterializationReport>,
    cache_provenance: CacheAccessProvenance,
}

fn validate_artifact(request: &CacheValidateRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    request.query.validate()?;
    if request.overlays.is_empty() {
        return Err(CliError::input(
            "cache validate requires at least one ordered remote overlay",
        ));
    }
    if matches!(request.mode, CacheValidationMode::Fast)
        && request.query.minimum_assurance == ArtifactAssuranceState::Certified
    {
        return Err(CliError::input(
            "certified cache consumption requires full dependency-closure validation",
        ));
    }
    match (
        request.mode,
        request.parts_root.as_ref(),
        request.package_destination.as_ref(),
        request.dependency_packages_root.as_ref(),
    ) {
        (CacheValidationMode::Fast, None, None, None) => {}
        (CacheValidationMode::Full, Some(parts), Some(package), Some(dependencies))
            if !parts.as_os_str().is_empty()
                && !package.as_os_str().is_empty()
                && !dependencies.as_os_str().is_empty() => {}
        (CacheValidationMode::Fast, _, _, _) => {
            return Err(CliError::input(
                "fast cache validation must not configure payload materialization paths",
            ));
        }
        (CacheValidationMode::Full, _, _, _) => {
            return Err(CliError::input(
                "full cache validation requires parts_root, package_destination, and dependency_packages_root",
            ));
        }
    }
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let resolution = resolve_remote_semantic_artifact(
        &remote,
        &cancellation,
        &request.query,
        &request.overlays,
    )?;
    let materialization = if matches!(request.mode, CacheValidationMode::Full) {
        let artifact = resolution
            .selected
            .as_ref()
            .ok_or_else(|| CliError::input("full cache validation requires a resolved artifact"))?;
        Some(materialize_resolved_remote_artifact_closure(
            &remote,
            artifact,
            request.parts_root.as_ref().expect("validated full request"),
            request
                .dependency_packages_root
                .as_ref()
                .expect("validated full request"),
            request
                .package_destination
                .as_ref()
                .expect("validated full request"),
            &request.transport.resources,
            &cancellation,
        )?)
    } else {
        None
    };
    let provenance = record_remote_cache_access(RemoteCacheAccessProvenanceRequest {
        operation: "cache.validate",
        family: &request.query.family,
        overlays: &request.overlays,
        resolution: &resolution,
        reuse_disposition: CacheReuseDisposition::InspectedOnly,
        validation_mode: match request.mode {
            CacheValidationMode::Fast => ProvenanceValidationMode::Fast,
            CacheValidationMode::Full => ProvenanceValidationMode::Full,
        },
        validation_outcome: if resolution.selected.is_some() {
            CacheValidationOutcome::Passed
        } else {
            CacheValidationOutcome::Failed
        },
        validation_detail: None,
        materialization: materialization.as_ref(),
    })?;
    write_success(
        "cache.validate",
        &CacheValidateCommandReport {
            mode: request.mode,
            resolution,
            materialization,
            cache_provenance: provenance,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheShardStatusRequest {
    schema_version: u32,
    family: String,
    visibility: CacheVisibility,
    topology_source: RemoteTopologySource,
    topology_trust: TopologyTrustPolicy,
    network: CacheNetworkRegistry,
    transport: GitReadTransportRequest,
}

#[derive(Debug, Serialize)]
struct CacheShardStatusEntry {
    shard_id: String,
    endpoint_id: String,
    status: xc_cache::TopologyShardStatus,
    repository: String,
    branch: String,
    revision: String,
    ledger: CapacityLedger,
    ledger_source: xc_cache::RemoteReadReport,
}

#[derive(Debug, Serialize)]
struct CacheShardStatusReport {
    family: String,
    visibility: CacheVisibility,
    topology_revision: String,
    topology_digest: xc_cache::ContentDigest,
    topology_source: xc_cache::RemoteReadReport,
    shards: Vec<CacheShardStatusEntry>,
}

fn shard_status(request: &CacheShardStatusRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    request.topology_source.validate()?;
    request.network.validate()?;
    if request.family.trim().is_empty()
        || !matches!(
            request.visibility,
            CacheVisibility::Private | CacheVisibility::Public
        )
    {
        return Err(CliError::input(
            "cache shard-status requires a family and private/public visibility",
        ));
    }
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let topology_revision = remote.read_ref(
        &request.topology_source.repository,
        &request.topology_source.branch,
    )?;
    let topology = RemoteShardReader::new(&remote, request.topology_source.maximum_registry_bytes)?
        .load_trusted_topology(
            &request.topology_source.repository,
            &topology_revision,
            &request.topology_source.registry_path,
            &request.topology_trust,
            &cancellation,
        )?;
    let route =
        xc_cache::resolve_topology_family(&topology.registry, &request.family, request.visibility)?;
    let ledger_reader = RemoteShardReader::new(
        &remote,
        request.topology_source.maximum_capacity_ledger_bytes,
    )?;
    let mut shards = Vec::with_capacity(route.ordered_shards.len());
    for shard in route.ordered_shards {
        let endpoint = request
            .network
            .endpoint_for_shard(&shard.endpoint_id)
            .ok_or_else(|| CliError::input("topology shard endpoint is absent from network"))?;
        if !endpoint.enabled_for_read || endpoint.visibility != request.visibility {
            return Err(CliError::input(
                "topology shard endpoint is not readable for the requested visibility",
            ));
        }
        let repository = endpoint.preferred_clone_url();
        let revision = remote.read_ref(&repository, &endpoint.branch)?;
        let ledger = ledger_reader.read_json::<CapacityLedger>(
            &repository,
            &revision,
            &request.topology_source.capacity_ledger_path,
            &cancellation,
        )?;
        ledger.value.validate()?;
        if ledger.value.shard_id != shard.shard_id {
            return Err(CliError::input(
                "capacity ledger shard identity does not match topology",
            ));
        }
        shards.push(CacheShardStatusEntry {
            shard_id: shard.shard_id,
            endpoint_id: shard.endpoint_id,
            status: shard.status,
            repository,
            branch: endpoint.branch.clone(),
            revision,
            ledger: ledger.value,
            ledger_source: ledger.source,
        });
    }
    write_success(
        "cache.shard-status",
        &CacheShardStatusReport {
            family: request.family.clone(),
            visibility: request.visibility,
            topology_revision,
            topology_digest: topology.topology_digest,
            topology_source: topology.source,
            shards,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheRevokeRequest {
    schema_version: u32,
    shard_id: String,
    network: CacheNetworkRegistry,
    record: RevocationRecord,
    maximum_partition_bytes: u64,
    execute_remote_mutations: bool,
    confirm_revocation: bool,
    transport: GitTransportRequest,
}

impl CacheRevokeRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        self.network.validate()?;
        self.record.validate()?;
        self.transport.validate()?;
        if self.shard_id.trim().is_empty() || self.maximum_partition_bytes == 0 {
            return Err(CliError::input(
                "cache revoke requires a shard_id and positive partition byte bound",
            ));
        }
        let endpoint = self
            .network
            .endpoint_for_shard(&self.shard_id)
            .ok_or_else(|| CliError::input("revocation shard is absent from network registry"))?;
        if !endpoint.enabled_for_read
            || (self.execute_remote_mutations && !endpoint.enabled_for_write)
        {
            return Err(CliError::input(
                "revocation endpoint lacks the required read/write capability",
            ));
        }
        match (self.execute_remote_mutations, self.confirm_revocation) {
            (false, false) | (true, true) => Ok(()),
            (false, true) => Err(CliError::input(
                "dry-run revocation requires confirm_revocation=false",
            )),
            (true, false) => Err(CliError::input(
                "remote revocation requires confirm_revocation=true",
            )),
        }
    }
}

#[derive(Debug, Serialize)]
struct CacheRevokeCommandReport {
    remote_mutation_enabled: bool,
    plan: RevocationUpdatePlan,
    execution: Option<RevocationUpdateOutcome>,
}

fn revoke_identity(request: &CacheRevokeRequest) -> Result<(), CliError> {
    request.validate()?;
    let endpoint = request
        .network
        .endpoint_for_shard(&request.shard_id)
        .expect("validated endpoint");
    let authorized_repository = format!("{}/{}", endpoint.owner, endpoint.repository);
    let repository = endpoint.preferred_clone_url();
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let plan = plan_revocation_update(
        &remote,
        &authorized_repository,
        &repository,
        &endpoint.branch,
        request.maximum_partition_bytes,
        request.record.clone(),
        &cancellation,
    )?;
    let execution = if request.execute_remote_mutations {
        let session =
            GitHubCredentialApiProbe::default().probe_repository(&authorized_repository)?;
        Some(execute_revocation_update(
            &remote,
            &session,
            &request.transport.staged_parts_root,
            &request.transport.resources,
            &cancellation,
            &plan,
        )?)
    } else {
        None
    };
    write_success(
        "cache.revoke",
        &CacheRevokeCommandReport {
            remote_mutation_enabled: request.execute_remote_mutations,
            plan,
            execution,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheSupersedeRequest {
    schema_version: u32,
    shard_id: String,
    network: CacheNetworkRegistry,
    record: SupersessionRecord,
    maximum_partition_bytes: u64,
    execute_remote_mutations: bool,
    confirm_supersession: bool,
    transport: GitTransportRequest,
}

impl CacheSupersedeRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        self.network.validate()?;
        self.record.validate()?;
        self.transport.validate()?;
        if self.shard_id.trim().is_empty() || self.maximum_partition_bytes == 0 {
            return Err(CliError::input(
                "cache supersede requires a shard_id and positive partition byte bound",
            ));
        }
        let endpoint = self
            .network
            .endpoint_for_shard(&self.shard_id)
            .ok_or_else(|| CliError::input("supersession shard is absent from network registry"))?;
        if !endpoint.enabled_for_read
            || (self.execute_remote_mutations && !endpoint.enabled_for_write)
        {
            return Err(CliError::input(
                "supersession endpoint lacks the required read/write capability",
            ));
        }
        match (self.execute_remote_mutations, self.confirm_supersession) {
            (false, false) | (true, true) => Ok(()),
            (false, true) => Err(CliError::input(
                "dry-run supersession requires confirm_supersession=false",
            )),
            (true, false) => Err(CliError::input(
                "remote supersession requires confirm_supersession=true",
            )),
        }
    }
}

#[derive(Debug, Serialize)]
struct CacheSupersedeCommandReport {
    remote_mutation_enabled: bool,
    plan: SupersessionUpdatePlan,
    execution: Option<SupersessionUpdateOutcome>,
}

fn supersede_artifact(request: &CacheSupersedeRequest) -> Result<(), CliError> {
    request.validate()?;
    let endpoint = request
        .network
        .endpoint_for_shard(&request.shard_id)
        .expect("validated endpoint");
    let authorized_repository = format!("{}/{}", endpoint.owner, endpoint.repository);
    let repository = endpoint.preferred_clone_url();
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let plan = plan_supersession_update(
        &remote,
        &authorized_repository,
        &repository,
        &endpoint.branch,
        request.maximum_partition_bytes,
        request.record.clone(),
        &cancellation,
    )?;
    let execution = if request.execute_remote_mutations {
        let session =
            GitHubCredentialApiProbe::default().probe_repository(&authorized_repository)?;
        Some(execute_supersession_update(
            &remote,
            &session,
            &request.transport.staged_parts_root,
            &request.transport.resources,
            &cancellation,
            &plan,
        )?)
    } else {
        None
    };
    write_success(
        "cache.supersede",
        &CacheSupersedeCommandReport {
            remote_mutation_enabled: request.execute_remote_mutations,
            plan,
            execution,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheRolloverRequest {
    schema_version: u32,
    authorized_registry_repository: String,
    registry_repository: String,
    registry_branch: String,
    registry_path: String,
    maximum_registry_bytes: u64,
    topology_trust: TopologyTrustPolicy,
    family: String,
    visibility: CacheVisibility,
    prior_writable_shard_id: String,
    successor_readiness: SuccessorShardReadinessEvidence,
    execute_remote_mutations: bool,
    confirm_rollover: bool,
    transport: GitTransportRequest,
}

impl CacheRolloverRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        self.transport.validate()?;
        self.successor_readiness.validate()?;
        if self.authorized_registry_repository.trim().is_empty()
            || self.registry_repository.trim().is_empty()
            || self.registry_branch.trim().is_empty()
            || self.registry_path.trim().is_empty()
            || self.family.trim().is_empty()
            || self.prior_writable_shard_id.trim().is_empty()
            || self.maximum_registry_bytes == 0
        {
            return Err(CliError::input(
                "cache rollover requires exact registry, family, shard, and read-bound inputs",
            ));
        }
        match (self.execute_remote_mutations, self.confirm_rollover) {
            (false, false) | (true, true) => Ok(()),
            (false, true) => Err(CliError::input(
                "dry-run rollover requires confirm_rollover=false",
            )),
            (true, false) => Err(CliError::input(
                "remote rollover requires confirm_rollover=true",
            )),
        }
    }
}

#[derive(Debug, Serialize)]
struct CacheRolloverCommandReport {
    remote_mutation_enabled: bool,
    plan: TopologyRolloverPlan,
    execution: Option<TopologyRolloverOutcome>,
}

fn rollover_shard(request: &CacheRolloverRequest) -> Result<(), CliError> {
    request.validate()?;
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let plan = plan_topology_rollover(
        &remote,
        &request.authorized_registry_repository,
        &request.registry_repository,
        &request.registry_branch,
        &request.registry_path,
        request.maximum_registry_bytes,
        &request.topology_trust,
        &request.family,
        request.visibility,
        &request.prior_writable_shard_id,
        &request.successor_readiness,
        &cancellation,
    )?;
    let execution = if request.execute_remote_mutations {
        let session = GitHubCredentialApiProbe::default()
            .probe_repository(&request.authorized_registry_repository)?;
        Some(execute_topology_rollover(
            &remote,
            &session,
            &request.transport.staged_parts_root,
            &request.transport.resources,
            &cancellation,
            &plan,
        )?)
    } else {
        None
    };
    write_success(
        "cache.rollover",
        &CacheRolloverCommandReport {
            remote_mutation_enabled: request.execute_remote_mutations,
            plan,
            execution,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheFetchRequest {
    schema_version: u32,
    query: RemoteSemanticQuery,
    overlays: Vec<RemoteResolverOverlay>,
    transport: GitReadTransportRequest,
    parts_root: PathBuf,
    package_destination: PathBuf,
}

#[derive(Debug, Serialize)]
struct CacheFetchCommandReport {
    resolution: SemanticResolutionReport,
    materialization: Option<RemoteArtifactMaterializationReport>,
    cache_provenance: CacheAccessProvenance,
}

fn fetch_artifact(request: &CacheFetchRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    request.query.validate()?;
    if request.overlays.is_empty()
        || request.parts_root.as_os_str().is_empty()
        || request.package_destination.as_os_str().is_empty()
    {
        return Err(CliError::input(
            "cache fetch requires an overlay, part-store root, and package destination",
        ));
    }
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let resolution = resolve_remote_semantic_artifact(
        &remote,
        &cancellation,
        &request.query,
        &request.overlays,
    )?;
    let materialization = resolution
        .selected
        .as_ref()
        .map(|artifact| {
            materialize_resolved_remote_artifact(
                &remote,
                artifact,
                &request.parts_root,
                &request.package_destination,
                &request.transport.resources,
                &cancellation,
            )
        })
        .transpose()?;
    let provenance = record_remote_cache_access(RemoteCacheAccessProvenanceRequest {
        operation: "cache.fetch",
        family: &request.query.family,
        overlays: &request.overlays,
        resolution: &resolution,
        reuse_disposition: if resolution.selected.is_some() {
            CacheReuseDisposition::Reused
        } else {
            CacheReuseDisposition::InspectedOnly
        },
        validation_mode: ProvenanceValidationMode::Full,
        validation_outcome: if materialization.is_some() {
            CacheValidationOutcome::Passed
        } else {
            CacheValidationOutcome::Failed
        },
        validation_detail: Some(
            "root artifact transport and canonical payload validation".to_owned(),
        ),
        materialization: None,
    })?;
    write_success(
        "cache.fetch",
        &CacheFetchCommandReport {
            resolution,
            materialization,
            cache_provenance: provenance,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheAuditRequest {
    schema_version: u32,
    repository: String,
    branch: String,
    shard_id: String,
    policy: ShardAuditPolicy,
    transport: GitReadTransportRequest,
}

fn audit_shard(request: &CacheAuditRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    request.policy.validate()?;
    if request.repository.trim().is_empty()
        || request.branch.trim().is_empty()
        || request.shard_id.trim().is_empty()
    {
        return Err(CliError::input(
            "cache audit requires repository, branch, and shard_id",
        ));
    }
    let remote = request.transport.open()?;
    let revision = remote.read_ref(&request.repository, &request.branch)?;
    let report = audit_remote_shard(
        &remote,
        &request.repository,
        &request.branch,
        &revision,
        &request.shard_id,
        &request.policy,
        &CancellationToken::for_policy(&request.transport.resources),
    )?;
    write_success("cache.audit", &report)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheIndexRepairRequest {
    schema_version: u32,
    shard_id: String,
    network: CacheNetworkRegistry,
    audit_policy: ShardAuditPolicy,
    repair_policy: ShardIndexRepairPolicy,
    execute_remote_mutations: bool,
    confirm_repair: bool,
    transport: GitTransportRequest,
}

impl CacheIndexRepairRequest {
    fn endpoint(&self) -> Result<&GitHubRepositoryEndpoint, CliError> {
        self.network
            .endpoint_for_shard(&self.shard_id)
            .ok_or_else(|| CliError::input("repair shard is absent from the network registry"))
    }

    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        self.network.validate()?;
        self.audit_policy.validate()?;
        self.repair_policy.validate()?;
        self.transport.validate()?;
        if self.shard_id.trim().is_empty() {
            return Err(CliError::input("cache repair-index requires shard_id"));
        }
        let endpoint = self.endpoint()?;
        if !endpoint.enabled_for_read
            || (self.execute_remote_mutations && !endpoint.enabled_for_write)
        {
            return Err(CliError::input(
                "repair shard endpoint lacks the required read or write capability",
            ));
        }
        match (self.execute_remote_mutations, self.confirm_repair) {
            (false, false) | (true, true) => Ok(()),
            (false, true) => Err(CliError::input(
                "a dry-run repair-index request must set confirm_repair=false",
            )),
            (true, false) => Err(CliError::input(
                "remote index repair requires confirm_repair=true",
            )),
        }
    }
}

#[derive(Serialize)]
struct CacheIndexRepairCommandReport {
    audit: RemoteShardAuditReport,
    plan: ShardIndexRepairPlan,
    execution: Option<ShardIndexRepairOutcome>,
}

fn repair_shard_index(request: &CacheIndexRepairRequest) -> Result<(), CliError> {
    request.validate()?;
    let endpoint = request.endpoint()?;
    let repository = endpoint.preferred_clone_url();
    let authorized_repository = format!("{}/{}", endpoint.owner, endpoint.repository);
    let remote = request.transport.open()?;
    let cancellation = CancellationToken::for_policy(&request.transport.resources);
    let revision = remote.read_ref(&repository, &endpoint.branch)?;
    let audit = audit_remote_shard(
        &remote,
        &repository,
        &endpoint.branch,
        &revision,
        &request.shard_id,
        &request.audit_policy,
        &cancellation,
    )?;
    let plan = plan_shard_index_repair(
        &remote,
        &audit,
        &authorized_repository,
        &request.repair_policy,
        &cancellation,
    )?;
    let execution = if request.execute_remote_mutations {
        let session =
            GitHubCredentialApiProbe::default().probe_repository(&authorized_repository)?;
        Some(execute_shard_index_repair(
            &remote,
            &session,
            &request.transport.staged_parts_root,
            &request.transport.resources,
            &cancellation,
            &plan,
        )?)
    } else {
        None
    };
    write_success(
        "cache.repair-index",
        &CacheIndexRepairCommandReport {
            audit,
            plan,
            execution,
        },
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CachePruneRequest {
    schema_version: u32,
    dry_run: bool,
    execute_local_deletion: bool,
    confirm_prune: bool,
    policy: LocalPrunePolicy,
    durability_policy: DurabilityPolicy,
    candidate: LocalPruneCandidate,
    copies: Vec<ArtifactCopyEvidence>,
    resources: ResourcePolicy,
}

fn plan_prune(request: &CachePruneRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    match (
        request.dry_run,
        request.execute_local_deletion,
        request.confirm_prune,
    ) {
        (true, false, false) => {
            let report = plan_local_prune(
                &request.policy,
                &request.durability_policy,
                &request.candidate,
                &request.copies,
            )?;
            write_success("cache.prune", &report)
        }
        (false, true, true) => {
            let report = execute_local_prune(
                &request.policy,
                &request.durability_policy,
                &request.candidate,
                &request.copies,
                &CancellationToken::for_policy(&request.resources),
            )?;
            write_success("cache.prune", &report)
        }
        _ => Err(CliError::input(
            "cache prune requires either an unconfirmed dry run or explicitly confirmed local deletion",
        )),
    }
}

fn plan_cache(request: &CachePlanRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    let report = plan_cache_derivations(request)?;
    write_success("cache.plan", &report)
}

fn plan_placement(request: &StoragePlacementRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    let report = plan_storage_placement(request)?;
    write_success("cache.placement-plan", &report)
}

fn plan_dedup(request: &DeduplicationPlanningRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    let report = plan_deduplication(request)?;
    write_success("cache.dedup-plan", &report)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheBundleExportCommandRequest {
    schema_version: u32,
    roots: Vec<CacheBundleArtifactIdentity>,
    sources: Vec<CacheBundleExportSource>,
    destination: PathBuf,
    policy: CacheBundlePolicy,
    resources: ResourcePolicy,
}

fn export_bundle(request: &CacheBundleExportCommandRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    let report = export_cache_bundle(
        &CacheBundleExportRequest {
            schema_version: request.schema_version,
            roots: request.roots.clone(),
            sources: request.sources.clone(),
        },
        &request.destination,
        &request.policy,
        &request.resources,
        &CancellationToken::for_policy(&request.resources),
    )?;
    write_success("cache.export", &report)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheBundleVerifyCommandRequest {
    schema_version: u32,
    bundle_root: PathBuf,
    scratch_root: PathBuf,
    policy: CacheBundlePolicy,
    resources: ResourcePolicy,
}

fn verify_bundle(request: &CacheBundleVerifyCommandRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    let report = verify_cache_bundle(
        &request.bundle_root,
        &request.scratch_root,
        &request.policy,
        &request.resources,
        &CancellationToken::for_policy(&request.resources),
    )?;
    write_success("cache.verify-bundle", &report)
}

#[derive(Serialize)]
struct LiveAcceptanceVerificationReport<'a> {
    valid: bool,
    acceptance_kind: &'static str,
    source_revision: &'a str,
    evidence_digest: &'a xc_cache::ContentDigest,
}

fn verify_live_acceptance(artifact: &LiveGitHubAcceptanceArtifact) -> Result<(), CliError> {
    verify_live_github_acceptance(artifact)?;
    let report = match artifact {
        LiveGitHubAcceptanceArtifact::Publication(record) => LiveAcceptanceVerificationReport {
            valid: true,
            acceptance_kind: "publication",
            source_revision: &record.source_revision,
            evidence_digest: &record.evidence_digest,
        },
        LiveGitHubAcceptanceArtifact::LargeCorpus(record) => LiveAcceptanceVerificationReport {
            valid: true,
            acceptance_kind: "large_corpus",
            source_revision: &record.source_revision,
            evidence_digest: &record.evidence_digest,
        },
    };
    write_success("cache.verify-live-acceptance", &report)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheBundleConsumeCommandRequest {
    schema_version: u32,
    bundle_root: PathBuf,
    identity: CacheBundleArtifactIdentity,
    destination: PathBuf,
    bundle_policy: CacheBundlePolicy,
    consumption_policy: CacheBundleConsumptionPolicy,
    resources: ResourcePolicy,
}

fn consume_bundle(request: &CacheBundleConsumeCommandRequest) -> Result<(), CliError> {
    validate_request_schema(request.schema_version)?;
    let report = materialize_cache_bundle_artifact(
        &request.bundle_root,
        &request.identity,
        &request.destination,
        &request.bundle_policy,
        &request.consumption_policy,
        &request.resources,
        &CancellationToken::for_policy(&request.resources),
    )?;
    write_success("cache.consume-bundle", &report)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicationAbandonRequest {
    schema_version: u32,
    journal_root: PathBuf,
    transaction_id: String,
    destination: PublicationDestination,
    reason: String,
    confirm_abandon: bool,
    transport: GitTransportRequest,
}

impl PublicationAbandonRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        validate_transaction_locator(&self.journal_root, &self.transaction_id)?;
        if !self.confirm_abandon {
            return Err(CliError::input(
                "publication abandonment requires confirm_abandon=true",
            ));
        }
        if self.reason.trim().is_empty() || self.reason.len() > 4_096 {
            return Err(CliError::input(
                "publication abandonment reason must contain 1 to 4096 bytes",
            ));
        }
        self.transport.validate()
    }
}

#[derive(Serialize)]
struct PublicationResumeCommandReport {
    remote_mutation_enabled: bool,
    steps_executed: u32,
    step_limit_reached: bool,
    recovery: xc_cache::PublicationRecoveryReport,
}

fn resume_publication(request: &PublicationResumeRequest) -> Result<(), CliError> {
    request.validate()?;
    let checkpoints = PublicationJournalStore::new(&request.journal_root);
    let mut journal = checkpoints.load_latest(&request.transaction_id)?;
    if !request.execute_remote_mutations {
        return write_success(
            "cache.resume",
            &PublicationResumeCommandReport {
                remote_mutation_enabled: false,
                steps_executed: 0,
                step_limit_reached: false,
                recovery: inspect_publication_recovery(&journal)?,
            },
        );
    }

    let execution = request
        .execution
        .as_ref()
        .expect("validated mutating resume has execution settings");
    validate_execution_targets(&journal, execution)?;
    let remote = request
        .transport
        .as_ref()
        .expect("validated mutating resume has transport settings")
        .open()?;
    let route_guard = request
        .route_guard
        .as_ref()
        .expect("validated mutating resume has route-guard settings");
    let probe = GitHubCredentialApiProbe::default();
    let mut sessions = BTreeMap::new();
    for (destination, target) in &journal.targets {
        if !matches!(
            target.state,
            PublicationTargetState::ReceiptComplete | PublicationTargetState::Abandoned
        ) {
            sessions.insert(
                *destination,
                probe.probe_repository(&target.authorized_repository)?,
            );
        }
    }
    let report = execute_publication_journal(
        &remote,
        &checkpoints,
        &mut journal,
        execution,
        &sessions,
        &PublicationRouteGuard {
            family: &route_guard.family,
            publication_policy: &route_guard.publication_policy,
            topology_source: &route_guard.topology_source,
            topology_trust: &route_guard.topology_trust,
            fabric_trust: &route_guard.fabric_trust,
            evaluation_unix_seconds: route_guard.evaluation_unix_seconds,
            network: &route_guard.network,
        },
    )?;
    write_success("cache.resume", &report)
}

fn validate_execution_targets(
    journal: &xc_cache::PublicationTransactionJournal,
    execution: &PublicationExecutionRequest,
) -> Result<(), CliError> {
    let active = journal
        .targets
        .iter()
        .filter_map(|(destination, target)| {
            (!matches!(
                target.state,
                PublicationTargetState::ReceiptComplete | PublicationTargetState::Abandoned
            ))
            .then_some(*destination)
        })
        .collect::<Vec<_>>();
    if active
        .iter()
        .any(|destination| !execution.targets.contains_key(destination))
        || execution
            .targets
            .keys()
            .any(|destination| !journal.targets.contains_key(destination))
    {
        return Err(CliError::input(
            "publication execution must configure every incomplete journal target and no unknown target",
        ));
    }
    Ok(())
}

fn validate_execution_policy(
    journal: &xc_cache::PublicationTransactionJournal,
    execution: &PublicationExecutionRequest,
    family: &str,
    policy: &CachePublicationPolicy,
) -> Result<(), CliError> {
    if policy.digest()? != journal.policy_digest {
        return Err(CliError::input(
            "resume publication policy does not match the journal policy digest",
        ));
    }
    for (destination, target) in &execution.targets {
        let bundle = &target.metadata_bundle;
        let required = policy.required_validators.get(destination);
        let missing_validator = required.into_iter().flatten().find(|required| {
            !bundle.validator_evidence.iter().any(|evidence| {
                evidence.validator_id == **required
                    && evidence.passed
                    && evidence.evidence_digest.validate()
            })
        });
        if bundle.family != family
            || policy
                .minimum_assurance
                .get(destination)
                .is_some_and(|minimum| bundle.achieved_assurance < *minimum)
            || missing_validator.is_some()
        {
            return Err(CliError::input(format!(
                "publication execution target {destination:?} does not satisfy the journal policy, family, assurance, or validator requirements"
            )));
        }
    }
    Ok(())
}

fn execute_publication_journal(
    remote: &GitCliRemoteStore,
    checkpoints: &PublicationJournalStore,
    journal: &mut xc_cache::PublicationTransactionJournal,
    execution: &PublicationExecutionRequest,
    sessions: &BTreeMap<PublicationDestination, xc_cache::AuthenticatedGitHubSession>,
    route_guard: &PublicationRouteGuard<'_>,
) -> Result<PublicationResumeCommandReport, CliError> {
    validate_execution_targets(journal, execution)?;
    validate_execution_policy(
        journal,
        execution,
        route_guard.family,
        route_guard.publication_policy,
    )?;
    let cancellation = CancellationToken::for_policy(&execution.resources);
    let mut steps_executed = 0u32;
    while steps_executed < execution.maximum_steps {
        let active = journal
            .targets
            .iter()
            .filter_map(|(destination, target)| {
                (!matches!(
                    target.state,
                    PublicationTargetState::ReceiptComplete | PublicationTargetState::Abandoned
                ))
                .then_some(*destination)
            })
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        for destination in active {
            if steps_executed == execution.maximum_steps {
                break;
            }
            validate_remote_publication_routes(
                remote,
                &cancellation,
                journal,
                route_guard.family,
                route_guard.topology_source,
                route_guard.topology_trust,
                route_guard.fabric_trust,
                route_guard.evaluation_unix_seconds,
                route_guard.network,
            )?;
            let target_request = &execution.targets[&destination];
            let target_execution = PublicationTargetExecution {
                staging_root: &execution.staging_root,
                resources: &execution.resources,
                finalization_policy: &target_request.finalization_policy,
                bundle: &target_request.metadata_bundle,
                public_sanitizer: (destination == PublicationDestination::Public)
                    .then_some(&route_guard.publication_policy.sanitizer),
                maximum_index_bytes: target_request.maximum_index_bytes,
                receipt_verified_at_unix_seconds: target_request.receipt_verified_at_unix_seconds,
                replace_existing_semantic: false,
            };
            advance_publication_target(
                remote,
                checkpoints,
                &cancellation,
                &sessions[&destination],
                journal,
                destination,
                &target_execution,
            )?;
            steps_executed += 1;
        }
    }
    let recovery = inspect_publication_recovery(journal)?;
    let step_limit_reached = steps_executed == execution.maximum_steps
        && recovery.targets.values().any(|target| {
            !matches!(
                target.state,
                PublicationTargetState::ReceiptComplete | PublicationTargetState::Abandoned
            )
        });
    Ok(PublicationResumeCommandReport {
        remote_mutation_enabled: true,
        steps_executed,
        step_limit_reached,
        recovery,
    })
}

fn abandon_publication(request: &PublicationAbandonRequest) -> Result<(), CliError> {
    request.validate()?;
    let checkpoints = PublicationJournalStore::new(&request.journal_root);
    let mut journal = checkpoints.load_latest(&request.transaction_id)?;
    let target = journal.targets.get(&request.destination).ok_or_else(|| {
        CliError::input(format!(
            "transaction has no {:?} publication target",
            request.destination
        ))
    })?;
    let session =
        GitHubCredentialApiProbe::default().probe_repository(&target.authorized_repository)?;
    let remote = request.transport.open()?;
    let report = abandon_publication_target(
        &remote,
        &checkpoints,
        &CancellationToken::for_policy(&request.transport.resources),
        &session,
        &mut journal,
        request.destination,
        &request.reason,
    )?;
    write_success("cache.abandon", &report)
}

fn validate_transaction_locator(journal_root: &Path, transaction_id: &str) -> Result<(), CliError> {
    if journal_root.as_os_str().is_empty()
        || transaction_id.len() != 64
        || !transaction_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CliError::input(
            "journal_root and a lowercase SHA-256 transaction_id are required",
        ));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageRequest {
    schema_version: u32,
    envelope: CanonicalPayloadEnvelope,
    sources: Vec<PayloadFileSource>,
    destination: PathBuf,
    resources: ResourcePolicy,
}

impl PackageRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        if self.sources.is_empty() || self.destination.as_os_str().is_empty() {
            return Err(CliError::input(
                "package request requires sources and a destination",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReconstructRequest {
    schema_version: u32,
    encoding: TransportEncodingRecord,
    parts_root: PathBuf,
    destination: PathBuf,
    resources: ResourcePolicy,
}

impl ReconstructRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        if self.parts_root.as_os_str().is_empty() || self.destination.as_os_str().is_empty() {
            return Err(CliError::input(
                "reconstruction request requires parts_root and destination",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifyPackageRequest {
    schema_version: u32,
    envelope: CanonicalPayloadEnvelope,
    encoding: TransportEncodingRecord,
    package_path: PathBuf,
    resources: ResourcePolicy,
}

impl VerifyPackageRequest {
    fn validate(&self) -> Result<(), CliError> {
        validate_request_schema(self.schema_version)?;
        if self.package_path.as_os_str().is_empty() {
            return Err(CliError::input(
                "verification request requires package_path",
            ));
        }
        Ok(())
    }
}

fn validate_request_schema(schema_version: u32) -> Result<(), CliError> {
    if schema_version != 1 {
        return Err(CliError::input(
            "command request schema_version must equal 1",
        ));
    }
    Ok(())
}

fn load_document<T: DeserializeOwned>(path: &Path) -> Result<T, CliError> {
    let context = |error: CliError| {
        error.with_artifact_context("request_load", "load command document", path)
    };
    let metadata = fs::symlink_metadata(path)
        .map_err(CliError::from)
        .map_err(context)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CliError::input(
            "command document must be a regular non-symlink file",
        ));
    }
    if metadata.len() > COMMAND_DOCUMENT_MAX_BYTES {
        return Err(CliError::input(
            "command document exceeds the 16 MiB safety limit",
        ));
    }
    let bytes = fs::read(path).map_err(CliError::from).map_err(context)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| CliError::input(error.to_string()))
        .map_err(context)
}

#[derive(Serialize)]
struct SuccessEnvelope<'a, T> {
    schema_version: u32,
    command: &'a str,
    status: &'static str,
    result: &'a T,
}

fn write_success<T: Serialize>(command: &str, result: &T) -> Result<(), CliError> {
    let output = SuccessEnvelope {
        schema_version: 1,
        command,
        status: "ok",
        result,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(CliError::from)?
    );
    Ok(())
}

#[derive(Debug)]
struct CliError {
    category: &'static str,
    diagnostic: Box<FailureDiagnostic>,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            category: "usage",
            diagnostic: Box::new(FailureDiagnostic::new(
                "command_dispatch",
                "parse command line",
                message,
                RetryClassification::AfterInputCorrection,
            )),
        }
    }

    fn input(message: impl Into<String>) -> Self {
        Self {
            category: "input",
            diagnostic: Box::new(FailureDiagnostic::new(
                "request_validation",
                "validate command request",
                message,
                RetryClassification::AfterInputCorrection,
            )),
        }
    }

    fn with_artifact_context(
        mut self,
        stage: impl Into<String>,
        operation: impl Into<String>,
        artifact: &Path,
    ) -> Self {
        self.diagnostic.stage = stage.into();
        self.diagnostic.operation = operation.into();
        self.diagnostic.artifact_identity = Some(artifact.to_string_lossy().into_owned());
        self
    }

    fn report(&self) -> ErrorEnvelope<'_> {
        ErrorEnvelope {
            schema_version: 1,
            status: "error",
            category: self.category,
            message: &self.diagnostic.source_cause,
            diagnostic: &self.diagnostic,
        }
    }

    fn exit_code(&self) -> i32 {
        if self.category == "usage" {
            2
        } else {
            1
        }
    }
}

impl Display for CliError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.diagnostic.source_cause)
    }
}

impl Error for CliError {}

impl From<CacheError> for CliError {
    fn from(error: CacheError) -> Self {
        let category = match &error {
            CacheError::Authentication(_) => "authentication",
            CacheError::PermissionDenied(_) => "permission_denied",
            CacheError::Cancelled(_) => "cancelled",
            CacheError::ResourceLimit(_) => "resource_limit",
            CacheError::DigestMismatch { .. } => "digest_mismatch",
            CacheError::NotFound(_) => "not_found",
            CacheError::InvalidManifest(_) | CacheError::InvalidTransition(_) => "invalid_state",
            CacheError::Io(_) | CacheError::Serialization(_) => "io",
            CacheError::ReadOnlyLayer(_) | CacheError::NoWritableShard(_) => "policy",
        };
        Self {
            category,
            diagnostic: Box::new(FailureDiagnostic::new(
                "cache_operation",
                "execute cache request",
                error.to_string(),
                match error {
                    CacheError::Authentication(_) | CacheError::PermissionDenied(_) => {
                        RetryClassification::AfterAuthorityChange
                    }
                    CacheError::Cancelled(_) => RetryClassification::ResumeFromCheckpoint,
                    CacheError::ResourceLimit(_) => RetryClassification::AfterResourceIncrease,
                    CacheError::Io(_) => RetryClassification::RetryUnchanged,
                    _ => RetryClassification::AfterInputCorrection,
                },
            )),
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(error: std::io::Error) -> Self {
        Self {
            category: "io",
            diagnostic: Box::new(FailureDiagnostic::new(
                "filesystem_io",
                "read or write command artifact",
                error.to_string(),
                RetryClassification::RetryUnchanged,
            )),
        }
    }
}

impl From<serde_json::Error> for CliError {
    fn from(error: serde_json::Error) -> Self {
        Self {
            category: "serialization",
            diagnostic: Box::new(FailureDiagnostic::new(
                "serialization",
                "serialize command document",
                error.to_string(),
                RetryClassification::AfterInputCorrection,
            )),
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    schema_version: u32,
    status: &'static str,
    category: &'a str,
    message: &'a str,
    diagnostic: &'a FailureDiagnostic,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_stable_cache_commands() {
        assert!(matches!(
            parse_command(&strings(&[
                "cache",
                "auth-probe",
                "example-org/public-shard"
            ]))
            .unwrap(),
            Command::AuthProbe { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "package", "request.json"])).unwrap(),
            Command::Package { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "transaction", "journals", "abc"])).unwrap(),
            Command::Transaction { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "resume", "request.json"])).unwrap(),
            Command::Resume { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "abandon", "request.json"])).unwrap(),
            Command::Abandon { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "find", "request.json"])).unwrap(),
            Command::Find { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "inspect", "request.json"])).unwrap(),
            Command::Inspect { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "validate", "request.json"])).unwrap(),
            Command::Validate { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "shard-status", "request.json"])).unwrap(),
            Command::ShardStatus { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "revoke", "request.json"])).unwrap(),
            Command::Revoke { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "supersede", "request.json"])).unwrap(),
            Command::Supersede { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "rollover", "request.json"])).unwrap(),
            Command::Rollover { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "fetch", "request.json"])).unwrap(),
            Command::Fetch { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "audit", "request.json"])).unwrap(),
            Command::Audit { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "repair-index", "request.json"])).unwrap(),
            Command::RepairIndex { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "prune", "request.json"])).unwrap(),
            Command::Prune { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "plan", "request.json"])).unwrap(),
            Command::Plan { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "placement-plan", "request.json"])).unwrap(),
            Command::PlacementPlan { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "dedup-plan", "request.json"])).unwrap(),
            Command::DedupPlan { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "export", "request.json"])).unwrap(),
            Command::Export { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "verify-bundle", "request.json"])).unwrap(),
            Command::VerifyBundle { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&[
                "cache",
                "verify-live-acceptance",
                "record.json"
            ]))
            .unwrap(),
            Command::VerifyLiveAcceptance { .. }
        ));
        assert!(matches!(
            parse_command(&strings(&["cache", "consume-bundle", "request.json"])).unwrap(),
            Command::ConsumeBundle { .. }
        ));
    }

    #[test]
    fn publish_command_requires_a_typed_request_document() {
        assert!(matches!(
            parse_command(&strings(&["cache", "publish", "request.json"])).unwrap(),
            Command::Publish { .. }
        ));
        assert!(parse_command(&strings(&["cache", "publish"])).is_err());
    }

    #[test]
    fn mutating_publication_binds_transport_and_execution_staging() {
        let transport = GitTransportRequest {
            temporary_root: PathBuf::from("temporary"),
            staged_parts_root: PathBuf::from("staging"),
            author_name: "Publisher".to_owned(),
            author_email: "publisher@example.invalid".to_owned(),
            resources: ResourcePolicy::default(),
        };
        let mut execution = PublicationExecutionRequest {
            staging_root: PathBuf::from("staging"),
            resources: ResourcePolicy::default(),
            maximum_steps: 1,
            targets: BTreeMap::new(),
        };
        assert!(execution.validate_transport_binding(&transport).is_ok());
        execution.staging_root = PathBuf::from("other-staging");
        assert!(execution.validate_transport_binding(&transport).is_err());
        execution.staging_root = PathBuf::from("staging");
        execution.resources.maximum_transfer_bytes = Some(1);
        assert!(execution.validate_transport_binding(&transport).is_err());
    }

    #[test]
    fn missing_request_reports_stage_artifact_and_retry_context() {
        let path = std::env::temp_dir().join(format!(
            "xc-missing-request-{}-diagnostic.json",
            std::process::id()
        ));
        let error = load_document::<serde_json::Value>(&path).unwrap_err();
        let report = serde_json::to_value(error.report()).unwrap();
        assert_eq!(report["diagnostic"]["stage"], "request_load");
        assert_eq!(report["diagnostic"]["operation"], "load command document");
        assert_eq!(
            report["diagnostic"]["artifact_identity"],
            path.to_string_lossy().as_ref()
        );
        assert_eq!(report["diagnostic"]["retry"], "retry_unchanged");
        assert_eq!(report["diagnostic"]["higher_precision_reasonable"], false);
    }
}
