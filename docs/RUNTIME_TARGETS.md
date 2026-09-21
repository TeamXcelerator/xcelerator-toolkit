# Runtime target profiles

`XC_TARGET_SPEC_FILE` names a JSON target specification. A complete specification
digest and external protocol-policy marker bind target-dependent identities. Changing the target or its
provider
creates new derived artifacts; original states and older measurements remain.

Schema 1 retains the existing Gaussian-polynomial evaluator and canonical digests.
Schema 3 delegates evaluation to an independently supplied executable. The Toolkit
does not include that executable or interpret its opaque input.

## External provider

A schema-3 specification has `schema_version`, an opaque `profile_id`, and
`external_profile` with these fields:

- `lambda_squared`: decimal cutoff squared, greater than one;
- `evaluation_precision_bits`: supported working precision, including guards;
- `provider_sha256`: SHA-256 of the authorized executable's exact bytes;
- `input`: provider-defined JSON data.

Set `XC_TARGET_PROVIDER_EXECUTABLE` to an absolute executable path. This is an
explicit authorization to run that program; a specification cannot choose an
executable or command. Use a trusted provider. The Toolkit checks its digest,
launches it with `--stdio`, and exchanges JSON lines over private pipes. The program
must write only protocol responses to stdout. Stderr is suppressed by default.
Neither the
input nor response values are included in protocol error messages.

The protocol version is 1. Every request has a monotonically increasing unsigned
`request_id` (starting at 1), `protocol_version: 1`, `operation`, and
`precision_bits`. Every reply must echo all four exactly. A missing, duplicated,
out-of-order, wrong-operation, or wrong-precision reply fails the connection.
Unversioned providers must be updated; they are not silently accepted.

For example, initialization at 256 working bits sends:

```json
{"protocol_version":1,"request_id":1,"operation":"initialize","precision_bits":256,"input":{}}
```

The provider replies with the same four envelope fields and `"ready":true`.
An evaluation request adds `"u":"1.25"`; its matching reply adds
`"value":"0.75"`. Decimal strings must retain the declared working precision.
These values are interface examples only. A reply containing `error` fails.
The precision field is a provider declaration, **not an independently verified
accuracy bound**. Exact values such as `0.5` need no redundant decimal digits.

Stdout is reserved for protocol frames. For private diagnostics, optionally set
`XC_TARGET_PROVIDER_STDERR_DIR` to an existing absolute directory. The Toolkit
creates a separate, non-overwriting log for each launched provider (mode 600 on
Unix; inherited directory permissions on Windows). These logs may contain private
values from the provider. Keep them outside publication inputs. Without this
explicit option, stderr is discarded. Logs are never copied into artifacts by
this interface.

Executable bytes are checked through an open file before launch and checked again
after launch. Windows retains a handle without write/delete sharing. On platforms
without that protection the second check detects ordinary replacement, but does
not give atomic execute-from-verified-bytes semantics against a hostile local
writer who can swap and restore the path. Use a trusted executable and directory.
Pinning a program cannot establish that it is deterministic, reads no external
state, or implements the intended mathematics; those are provider obligations.

The evaluator normalizes supplied values at u=1. A zero or nonfinite normalizer,
nonfinite result, cutoff mismatch, insufficient precision, malformed response,
or timeout is an error. Responses are limited to 1 MiB. The response timeout
defaults to 300 seconds and can be set to 1-3600 seconds through
`XC_TARGET_PROVIDER_TIMEOUT_SECONDS`. Keep the executable and input unchanged for
the duration of a run. A precision declaration is not an approximation certificate.

Both schemas support the existing optional auxiliary profile, whose scientific
interpretation remains the caller's responsibility. Schema 2 is no longer accepted;
prepare a new schema-3 input with the external provider. Existing target-dependent
artifacts keep their original identities and are not relabeled or replaced.

Distance entry points check the cutoff before warm reuse. Target-independent
matrices, eigenstates, and roots remain reusable. Runtime inputs and provider
implementations are not copied into artifacts by this interface.

This interface supports target distances and residual diagnostics. It does not
supply transform jets, atom tables, or proof hypotheses; those require separately
identified [external reference inputs](ULTRA_COMPLETENESS.md).

`profile_id` is part of the target identity. Renaming it also creates new
target-dependent identities. Protocol envelopes and precision declarations do
not upgrade point measurements to certified distances.

The versioned external protocol has a distinct digest domain. Earlier external
provider results remain retained but are not reused as measurements validated
under this protocol. Schema-1 digests and target-independent identities are
unchanged. No historical measurement is relabeled or deleted.

The native and HP evaluators expose `try_value` for callers that need the
underlying error. Built-in distance, crossing and residual paths preserve
per-point provider errors instead of reporting only the resulting nonfinite
arithmetic. The scalar `value` compatibility method returns NaN on failure.

Each evaluator construction starts a provider, verifies its executable before
and after launch, and initializes its state. Refinement rules may construct
separate evaluators; no process-global provider cache is used. With diagnostic
logging enabled, each construction creates a separate file, and a failed launch
may leave an empty file. Logs are not rotated automatically. Enable the setting
for a bounded troubleshooting session and manage the private directory afterward.
