#!/usr/bin/env python3
"""Index retained report packets or compact summaries; query exact decimal strings.
This is a derived navigation index, not numerical verification. No network access.
"""
import argparse, hashlib, json, re, sqlite3
from pathlib import Path
DECIMAL = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
def parse_field_policy(text):
    rules = []
    for index, line in enumerate(text.splitlines(), 1):
        parts = line.split(':', 1)
        if (len(parts) != 2 or parts[0] not in ('prefix', 'indexed') or not parts[1]
                or not re.fullmatch(r'[A-Za-z0-9_]+', parts[1]) or tuple(parts) in rules):
            raise ValueError(f'invalid compact field policy at line {index}')
        rules.append(tuple(parts))
    if not rules:
        raise ValueError('invalid compact field policy: empty policy')
    return rules

FIELD_POLICY = parse_field_policy((Path(__file__).resolve().parents[1] / 'crates/xc-spectral/src/ccm/compact_field_policy.txt').read_text(encoding='utf-8'))

def scalar_field(name):
    for kind, prefix in FIELD_POLICY:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if kind == 'prefix' or (suffix and all('0' <= c <= '9' for c in suffix)):
                return False
    return True


def rows(path, packet):
    report = packet.get('report', packet)
    data = report.get('data', report)
    if not isinstance(data, dict):
        return
    kind = report.get('kind', packet.get('manifest', {}).get('key', {}).get('kind', 'unclassified'))
    manifest = packet.get('manifest') or packet.get('source_manifest') or {}
    artifact = manifest.get('content_digest', packet.get('artifact_digest'))
    source = data.get('source') or report.get('source_dependencies', [])
    base = dict(C=data.get('lambda_squared'), C_definition='lambda_squared', N=data.get('n_modes'),
                P=data.get('source_precision_bits', data.get('precision_bits')), P_unit='bits',
                kind=kind, artifact_digest=artifact, sources=source, outcome=data.get('outcome', 'unassessed'),
                convention=data.get('convention', data.get('ordinal_scope', 'see source report')),
                units='not separately declared; see observable and convention', packet=str(path.resolve()))
    maps = [(None, 'aggregate', data.get('values', {}), base['outcome'], [])]
    maps += [(r.get('ordinal'), r.get('label', 'row'), r.get('values', {}), r.get('outcome', base['outcome']), r.get('notes', [])) for r in data.get('rows', []) if isinstance(r, dict)]
    if not data.get('values') and not data.get('rows'):
        maps.append((None, 'retained_field', data, base['outcome'], []))
    for ordinal, scope, values, outcome, notes in maps:
        for name, value in values.items():
            if isinstance(value, str) and DECIMAL.fullmatch(value) and scalar_field(name):
                yield dict(base, ordinal=ordinal, ordinal_scope=scope, observable=name, value=value, outcome=outcome, notes=notes)

def build(inputs, database, maximum_bytes=64 << 20):
    if database.exists():
        raise ValueError('Output database already exists; choose a new filename.')
    connection = sqlite3.connect(database)
    connection.execute('create table measurements (C text,N integer,P integer,ordinal integer,kind text,observable text,outcome text,value text,record text)')
    connection.execute('create table inventory (path text,sha256 text,status text,reason text)')
    connection.execute('create table metadata (key text primary key,value text)')
    connection.execute("insert into metadata values ('status','building')")
    connection.commit()
    count = 0
    paths = sorted(set(p for source in inputs for p in (source.rglob('*.json') if source.is_dir() else [source])))
    try:
        for path in paths:
            digest = None
            try:
                if path.stat().st_size > maximum_bytes:
                    raise ValueError('input byte limit exceeded; use compact capture summaries')
                raw = path.read_bytes()
                if len(raw) > maximum_bytes:
                    raise ValueError('input grew past byte limit')
                digest = hashlib.sha256(raw).hexdigest()
                packet = json.loads(raw)
                if not isinstance(packet, dict):
                    raise ValueError('not an object report')
                n = 0
                for row in rows(path, packet):
                    row['packet_sha256'] = digest
                    connection.execute('insert into measurements values (?,?,?,?,?,?,?,?,?)', tuple(row[k] for k in ('C','N','P','ordinal','kind','observable','outcome','value')) + (json.dumps(row, separators=(',',':')),))
                    n += 1
                count += n
                connection.execute('insert into inventory values (?,?,?,?)',(str(path),digest,'indexed' if n else 'unassessed',None if n else 'no recognized scalar rows'))
            except (OSError, ValueError, TypeError, AttributeError) as error:
                connection.execute('insert into inventory values (?,?,?,?)',(str(path),digest,'unassessed',str(error)))
        connection.execute('create index lookup on measurements (C,N,P,ordinal,observable)')
        connection.execute("update metadata set value='complete' where key='status'")
        connection.commit()
    finally:
        connection.close()
    return count

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    b=commands.add_parser('build');b.add_argument('inputs',nargs='+',type=Path);b.add_argument('--output',required=True,type=Path);b.add_argument('--maximum-input-bytes',type=int,default=64<<20)
    q=commands.add_parser('query');q.add_argument('database',type=Path)
    for name in ['C','N','P','ordinal','kind','observable','outcome']:q.add_argument('--'+name)
    q.add_argument('--limit',type=int,default=200)
    q.add_argument('--inventory',action='store_true')
    a=parser.parse_args()
    if a.command=='build':
        if a.maximum_input_bytes<=0:parser.error('byte limit must be positive')
        print(json.dumps(dict(indexed_measurements=build(a.inputs,a.output,a.maximum_input_bytes),database=str(a.output),scope='derived navigation; no numerical validation')))
    else:
        if a.limit<=0:parser.error('limit must be positive')
        db=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
        if db.execute("select value from metadata where key='status'").fetchone()!=('complete',):
            db.close();parser.error('index construction was not completed')
        if a.inventory:
            for row in db.execute('select path,sha256,status,reason from inventory order by path limit ?',(a.limit,)):print(json.dumps(dict(zip(['path','sha256','status','reason'],row))))
        else:
            fields=[name for name in ['C','N','P','ordinal','kind','observable','outcome'] if getattr(a,name) is not None]
            sql='select record from measurements'+(' where '+' and '.join(name+'=?' for name in fields) if fields else '')+' order by C,N,P,ordinal,observable limit ?'
            for (row,) in db.execute(sql,[getattr(a,name) for name in fields]+[a.limit]):print(row)
        db.close()
if __name__=='__main__':main()
