# Colab model loader for arc-agi-3-final-solution.ipynb (Colab kernel side).
# Executed by colab_adapter.py into its own namespace (_ARC3_HELPERS). It reads the
# main checkpoint from Drive via the Drive API (8 parallel connections), prepares the
# pinned Python 3.12 + SGLang environment from the Pennyroyal wheelhouse, and starts
# the model server. Carried over from the base notebook; only an unused harness-patch
# downloader was removed.
# Full-model loading trial. Keep G4 and do not reset the current runtime.
KAGGLE_SECRET_NAME = 'kaggle-key'
KAGGLE_AUTH_MODE = 'auto'
KAGGLE_USERNAME = ''
KAGGLE_USERNAME_SECRET_NAME = 'kaggle-username'
MOUNT_TIMEOUT_SECONDS = 60
ALLOW_DRAFT_DOWNLOAD_FALLBACK = True
MAX_DRAFT_DOWNLOAD_BYTES = 8*1024**3
DRAFT_DOWNLOAD_TIMEOUT_SECONDS = 20*60
PYTHON312 = '/usr/bin/python3.12'
SETUP_ROOT = '/content/arc3-runtime/python312-v1'
MIN_INSTALL_START_FREE_GIB = 30
MIN_LOCAL_FREE_GIB = 15
COMMAND_TIMEOUT_SECONDS = 1800
ALLOW_BOUNDED_SOURCE_DOWNLOAD = False
MAX_SOURCE_BYTES = 512*1024**2
RESOURCES = [
    {'name':'draft_model','kind':'model','handle':'dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'},
    {'name':'wheelhouse','kind':'dataset','handle':'dfranzen/pennyroyal-v253','version':1},
    {'name':'source_bundle','kind':'dataset','handle':'dfranzen/taaf-kaggle-source-bundle-copy','version':1},
]
SERVING_HOST = '127.0.0.1'
SERVED_MODEL_PORT = 8001
WEIGHT_DOWNLOAD_CONNECTIONS = 8
SHARD_DOWNLOAD_TIMEOUT_SECONDS = 10*60
MODEL_STARTUP_TIMEOUT_SECONDS = 60*60
MIN_AVAILABLE_RAM_GIB = 12
MIN_START_AVAILABLE_RAM_GIB = 140
REPORT_PATH = '/content/arc3-preflight/model_load_report.json'
DOWNLOAD_REPORT_TO_COMPUTER = True

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
MAX_RESPONSE_BYTES = 64 * 1024

def disk_snapshot(path='/content'):
    usage = shutil.disk_usage(path if Path(path).exists() else '/')
    return {name + '_gib': round(getattr(usage, name) / 1024 ** 3, 3) for name in ('total', 'used', 'free')}

def environment_snapshot():
    mem = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            name, value = line.split(':', 1)
            if name in {'MemTotal', 'MemAvailable'}:
                mem[name] = round(int(value.split()[0]) * 1024 / 1024 ** 3, 3)
    except (OSError, ValueError):
        pass
    gpu = []
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {'python': sys.version.split()[0], 'disk': disk_snapshot(), 'ram_gib': mem, 'gpu_name_vram_mib_driver': gpu, 'colab_mount_backend_present': bool(os.environ.get('TBE_RUNTIME_ADDR'))}

def resource_payload(resource):
    parts = resource['handle'].split('/')
    if resource['kind'] == 'model':
        if len(parts) not in {4, 5} or any((not part for part in parts)):
            raise ValueError('Model handle must be owner/model/framework/variation[/version].')
        payload = dict(zip(('owner', 'model', 'framework', 'variation'), parts[:4]))
        version = int(parts[4]) if len(parts) == 5 else None
    elif resource['kind'] == 'dataset':
        if len(parts) != 2 or any((not part for part in parts)):
            raise ValueError('Dataset handle must be owner/dataset; use the separate version field.')
        payload = dict(zip(('owner', 'dataset'), parts))
        version = resource.get('version')
    else:
        raise ValueError('Only model/dataset mounting is probed; competition data is a separate step.')
    if version is not None:
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError('Version must be a positive integer or None.')
        payload['version'] = version
    return payload

def authenticated_session():
    from google.colab import userdata
    if KAGGLE_AUTH_MODE not in {'auto', 'token', 'legacy', 'json'}:
        raise ValueError('KAGGLE_AUTH_MODE must be auto, token, legacy or json.')
    try:
        secret = userdata.get(KAGGLE_SECRET_NAME).strip()
    except Exception:
        raise RuntimeError(f'Enable notebook access for Colab Secret {KAGGLE_SECRET_NAME!r}.') from None
    if not secret:
        raise RuntimeError('The Kaggle Secret is empty.')
    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json'})
    session.trust_env = False
    if KAGGLE_AUTH_MODE == 'json' or (KAGGLE_AUTH_MODE == 'auto' and secret.startswith('{')):
        try:
            credentials = json.loads(secret)
            if not isinstance(credentials, dict) or not all((isinstance(credentials.get(k), str) and credentials[k].strip() for k in ('username', 'key'))):
                raise ValueError()
        except (ValueError, TypeError):
            session.close()
            raise RuntimeError('JSON Secret must contain non-empty username and key strings.') from None
        session.auth = (credentials['username'].strip(), credentials['key'].strip())
        mode = 'legacy_json'
    else:
        username = KAGGLE_USERNAME.strip()
        use_legacy = KAGGLE_AUTH_MODE == 'legacy'
        if KAGGLE_AUTH_MODE == 'auto' and (not secret.startswith('KGAT_')):
            use_legacy = bool(username or re.fullmatch('[0-9a-fA-F]{32}', secret))
            if not username:
                try:
                    username = userdata.get(KAGGLE_USERNAME_SECRET_NAME).strip()
                    use_legacy = bool(username) or use_legacy
                except Exception:
                    pass
        elif use_legacy and (not username):
            try:
                username = userdata.get(KAGGLE_USERNAME_SECRET_NAME).strip()
            except Exception:
                pass
        if use_legacy:
            if not username:
                session.close()
                raise RuntimeError('Legacy raw key requires KAGGLE_USERNAME or a kaggle-username Secret. If this is a modern token, select token mode.')
            session.auth = (username, secret)
            mode = 'legacy_key'
        else:
            session.headers['Authorization'] = 'Bearer ' + secret
            mode = 'api_token'
    return (session, mode)

def endpoint_call(session, resource, operation):
    """Bounded metadata requests only; never follows redirects or falls back to downloads."""
    if operation not in {'is_supported', 'mount'}:
        raise ValueError('Only support and mount operations are allowed.')
    backend = os.environ.get('TBE_RUNTIME_ADDR', '').strip()
    if not backend:
        return ({'status': 'BACKEND_UNAVAILABLE', 'http_status': None}, None)
    if not re.fullmatch('[A-Za-z0-9_.-]+(?::[0-9]+)?', backend):
        return ({'status': 'BACKEND_ADDRESS_UNRECOGNIZED', 'http_status': None}, None)
    try:
        payload = resource_payload(resource)
    except (ValueError, TypeError, KeyError):
        return ({'status': 'INVALID_RESOURCE_CONFIG', 'http_status': None}, None)
    endpoint = '/kagglehub/' + ('models' if resource['kind'] == 'model' else 'datasets') + '/' + operation
    try:
        with session.post('http://' + backend + endpoint, json=payload, timeout=(5, 15), allow_redirects=False, stream=True) as response:
            code = response.status_code
            details = {'http_status': code}
            if code == 401:
                return (dict(details, status='AUTH_FAILED'), None)
            if code == 403:
                return (dict(details, status='ACCESS_DENIED'), None)
            if code == 404:
                return (dict(details, status='NOT_SUPPORTED_OR_NOT_FOUND'), None)
            if 300 <= code < 400:
                return (dict(details, status='REDIRECT_BLOCKED'), None)
            if not 200 <= code < 300:
                return (dict(details, status='HTTP_ERROR'), None)
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=4096):
                raw.extend(chunk)
                if len(raw) > MAX_RESPONSE_BYTES:
                    return (dict(details, status='RESPONSE_TOO_LARGE'), None)
            data = None
            if raw:
                try:
                    data = json.loads(raw)
                except (ValueError, UnicodeError):
                    return (dict(details, status='UNEXPECTED_RESPONSE'), None)
            explicit_false = data is False or (isinstance(data, dict) and any((data.get(key) is False for key in ('supported', 'is_supported', 'isSupported'))))
            if operation == 'is_supported':
                return (dict(details, status='NOT_SUPPORTED' if explicit_false else 'SUPPORTED'), None)
            if not isinstance(data, dict) or not isinstance(data.get('slug'), str) or (not data['slug']):
                return (dict(details, status='MOUNT_RESPONSE_INVALID'), None)
            slug = Path(data['slug'])
            if slug.is_absolute() or '..' in slug.parts:
                return (dict(details, status='MOUNT_PATH_INVALID'), None)
            root = Path(os.environ.get('COLAB_CACHE_MOUNT_FOLDER', '/kaggle/input')) / slug
            return (dict(details, status='MOUNT_REQUEST_ACCEPTED'), root)
    except requests.Timeout:
        return ({'status': 'TIMEOUT', 'http_status': None}, None)
    except requests.ConnectionError:
        return ({'status': 'CONNECTION_ERROR', 'http_status': None}, None)
    except requests.RequestException:
        return ({'status': 'REQUEST_ERROR', 'http_status': None}, None)

def inspect_model_metadata(root, resource):
    """Read only config/index JSON and stat shard files; never reads tensor contents."""
    root = Path(root)
    configs = [root / 'config.json'] if (root / 'config.json').is_file() else []
    if resource['name'] == 'draft_model':
        for path in root.rglob('config.json'):
            if path not in configs:
                configs.append(path)
            if len(configs) >= 20:
                break
    checkpoints = []
    for config_path in configs:
        index_path = config_path.parent / 'model.safetensors.index.json'
        if not index_path.is_file():
            continue
        if max(config_path.stat().st_size, index_path.stat().st_size) > 16 * 1024 ** 2:
            checkpoints.append({'path': str(config_path.parent), 'status': 'METADATA_TOO_LARGE'})
            continue
        try:
            config = json.loads(config_path.read_text())
            index = json.loads(index_path.read_text())
            weight_map = index.get('weight_map')
            if not isinstance(config, dict) or not isinstance(weight_map, dict) or (not weight_map):
                raise ValueError()
            names = sorted(set(weight_map.values()))
            missing, bad, size = ([], [], 0)
            for name in names:
                if not isinstance(name, str):
                    bad.append('non-string shard path')
                    continue
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    bad.append(name)
                    continue
                shard = index_path.parent / relative
                if not shard.is_file() or shard.stat().st_size < 8:
                    missing.append(name)
                else:
                    size += shard.stat().st_size
            checkpoints.append({'path': str(config_path.parent), 'status': 'LAYOUT_OK' if not missing and (not bad) else 'LAYOUT_INCOMPLETE', 'shard_count': len(names), 'apparent_weights_gib': round(size / 1024 ** 3, 3), 'missing_shards': missing[:20], 'invalid_paths': bad[:20]})
        except (ValueError, TypeError, AttributeError, OSError):
            checkpoints.append({'path': str(config_path.parent), 'status': 'METADATA_ERROR'})
    return {'checkpoints': checkpoints, 'tensor_contents_read': False, 'note': 'Layout/stat check only; does not establish GPU compatibility or mounted-filesystem caching behavior.'}

def mount_auxiliary_resource(session, resource):
    outcome, _ = endpoint_call(session, resource, 'is_supported')
    record = dict(resource, support=outcome)
    if outcome['status'] != 'SUPPORTED':
        return record
    before = disk_snapshot()
    outcome, root = endpoint_call(session, resource, 'mount')
    record['mount'] = outcome
    if root is not None:
        deadline = time.monotonic() + MOUNT_TIMEOUT_SECONDS
        while not root.is_dir() and time.monotonic() < deadline:
            time.sleep(1)
        if root.is_dir():
            outcome.update(status='MOUNTED', path=str(root))
            try:
                if resource['kind'] == 'model':
                    outcome['metadata'] = inspect_model_metadata(root, resource)
                else:
                    entries = []
                    for index, path in enumerate(root.iterdir()):
                        if index >= 100:
                            break
                        entries.append({'name': path.name, 'is_directory': path.is_dir()})
                    outcome['top_level_entries'] = entries
                    outcome['file_contents_read'] = False
            except OSError:
                outcome['inspection_status'] = 'FILESYSTEM_ERROR'
        else:
            outcome['status'] = 'MOUNT_WAIT_TIMEOUT'
    after = disk_snapshot()
    outcome.update(disk_before=before, disk_after=after,
                   local_used_delta_gib=round(after['used_gib'] - before['used_gib'], 3))
    return record

import zipfile
import stat
import tempfile
import hashlib
from urllib.parse import urlparse, urljoin
class TransferFault(RuntimeError):
    def __init__(self, code, http_status=None):
        super().__init__(code)
        self.code, self.http_status = code, http_status

def safe_download_url(url):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    allowed = host == 'storage.googleapis.com' or host.endswith('.googleapis.com') or host.endswith('.kaggleusercontent.com') or host in {'www.kaggle.com', 'api.kaggle.com'}
    if not allowed or parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise TransferFault('DOWNLOAD_HOST_NOT_ALLOWED')
    return url

def public_get(session, url, **kwargs):
    # A credential-free session is deliberately used for signed download URLs.
    for _ in range(4):
        response = session.get(safe_download_url(url), allow_redirects=False, stream=True,
                               timeout=(10, 60), **kwargs)
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get('Location', '')
            response.close()
            url = urljoin(url, location)
            continue
        return response
    raise TransferFault('DOWNLOAD_REDIRECT_LIMIT')

from contextlib import contextmanager


def dataset_rpc(session,method,payload):
    if method not in {'ListDatasetFiles','DownloadDataset'}:raise ValueError('Unsupported RPC')
    return session.post('https://api.kaggle.com/v1/datasets.DatasetApiService/'+method,
                        json=payload,stream=True,allow_redirects=False,timeout=(10,60))


def source_rpc(session,method,payload,state,persist):
    for attempt in range(1,4):
        try:response=dataset_rpc(session,method,payload)
        except (requests.Timeout,requests.ConnectionError):
            if attempt==3:raise TransferFault('SOURCE_NETWORK_ERROR') from None
            print(f'Retrying source bundle connection: {attempt}/3',flush=True)
            time.sleep(2**attempt);continue
        state.update(method=method,last_http_status=response.status_code,attempt=attempt);persist()
        if response.status_code==429 or 500<=response.status_code<600:
            status=response.status_code;response.close()
            if attempt==3:raise TransferFault('SOURCE_API_HTTP',status)
            print(f'Retrying source bundle request: HTTP {status} | {attempt}/3',flush=True)
            time.sleep(2**attempt);continue
        return response


def source_json(response):
    raw=bytearray()
    for chunk in response.iter_content(8192):
        raw.extend(chunk)
        if len(raw)>2*1024**2:raise TransferFault('SOURCE_LIST_RESPONSE_TOO_LARGE')
    try:return json.loads(raw)
    except (ValueError,UnicodeError):raise TransferFault('SOURCE_JSON_INVALID') from None


def source_inventory(session,state=None,persist=None):
    state={} if state is None else state
    persist=(lambda:None) if persist is None else persist
    payload={'ownerSlug':'dfranzen','datasetSlug':'taaf-kaggle-source-bundle-copy',
             'datasetVersionNumber':1,'pageSize':1000}
    listing,tokens={},set()
    omitted=[]
    for page in range(40):
        with source_rpc(session,'ListDatasetFiles',payload,state,persist) as response:
            if response.status_code!=200:raise TransferFault('SOURCE_LIST_HTTP',response.status_code)
            data=source_json(response)
        if not isinstance(data,dict):raise TransferFault('SOURCE_LIST_INVALID')
        entries=data.get('datasetFiles',data.get('files'))
        if not isinstance(entries,list) or data.get('errorMessage'):raise TransferFault('SOURCE_LIST_INVALID')
        for index,entry in enumerate(entries):
            if not isinstance(entry,dict):
                state['invalid_entry']={'page':page+1,'index':index,'entry_type':type(entry).__name__};persist()
                raise TransferFault('SOURCE_FILE_INVALID')
            name=entry.get('name')
            # A missing/null totalBytes is unknown, never silently assumed to be zero.
            # Exact filenames remain mandatory; ZIP metadata and CRC will resolve the size.
            size=entry.get('totalBytes')
            if isinstance(size,str) and size.isdigit():size=int(size)
            valid_size=size is None or isinstance(size,int) and not isinstance(size,bool) and 0<=size<=MAX_SOURCE_BYTES
            if not isinstance(name,str) or not name or not valid_size:
                state['invalid_entry']={'page':page+1,'index':index,'name':name[:200] if isinstance(name,str) else None,
                    'fields':sorted(entry),'size_type':type(size).__name__};persist()
                print('Invalid source bundle file entry:',state['invalid_entry'],flush=True)
                raise TransferFault('SOURCE_FILE_INVALID')
            relative=Path(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name or '\x00' in name
                    or str(relative)!=name or name in listing):raise TransferFault('SOURCE_PATH_INVALID')
            listing[name]=size
            if size is None:omitted.append(name)
        if len(listing)>20000 or sum(size for size in listing.values() if size is not None)>MAX_SOURCE_BYTES:
            raise TransferFault('SOURCE_SIZE_BOUND_EXCEEDED')
        token=data.get('nextPageToken','')
        if not token:
            if not listing:raise TransferFault('SOURCE_LIST_EMPTY')
            state.update(file_count=len(listing),api_sizes_omitted=len(omitted),
                omitted_size_samples=omitted[:12],known_bytes=sum(size for size in listing.values() if size is not None))
            persist()
            if omitted:
                print(f'Source bundle: {len(omitted)} files have no API size field; sizes will be verified from the ZIP table.',flush=True)
            return listing
        if not isinstance(token,str) or token in tokens:raise TransferFault('SOURCE_PAGINATION_INVALID')
        tokens.add(token);payload['pageToken']=token
    raise TransferFault('SOURCE_PAGE_BOUND_EXCEEDED')


@contextmanager
def source_archive_stream(session,public,state,persist):
    payload={'ownerSlug':'dfranzen','datasetSlug':'taaf-kaggle-source-bundle-copy','datasetVersionNumber':1}
    with source_rpc(session,'DownloadDataset',payload,state,persist) as response:
        status=response.status_code;ctype=response.headers.get('Content-Type','').lower()
        if status in {301,302,303,307,308}:
            location=response.headers.get('Location','')
            if not location:raise TransferFault('SOURCE_REDIRECT_MISSING',status)
            url=urljoin('https://api.kaggle.com',location)
        elif status==200 and 'json' in ctype:
            data=source_json(response);url=data.get('url') if isinstance(data,dict) else None
            if not isinstance(url,str) or not url:
                code=data.get('code') if isinstance(data,dict) else None
                raise TransferFault('SOURCE_DOWNLOAD_JSON_ERROR',code if isinstance(code,int) else status)
        elif status==200 and 'html' not in ctype:
            state['transport']='DIRECT_API_STREAM';persist()
            yield response.iter_content(1024**2);return
        else:raise TransferFault('SOURCE_DOWNLOAD_HTTP',status)
        with public_get(public,url,headers={'Accept-Encoding':'identity'}) as media:
            state.update(transport='SIGNED_URL_STREAM',media_http_status=media.status_code);persist()
            if media.status_code!=200:raise TransferFault('SOURCE_SIGNED_DOWNLOAD_HTTP',media.status_code)
            if 'html' in media.headers.get('Content-Type','').lower():raise TransferFault('SOURCE_MEDIA_HTML')
            yield media.iter_content(1024**2)


def extract_source_zip(archive,expected,root):
    root=Path(root);resolved={};hashes={}
    with zipfile.ZipFile(archive) as z:
        members=[member for member in z.infolist() if not member.is_dir()]
        names=[member.filename for member in members]
        if len(names)!=len(set(names)) or set(names)!=set(expected):
            raise TransferFault('SOURCE_ARCHIVE_MANIFEST_MISMATCH')
        if sum(member.file_size for member in members)>MAX_SOURCE_BYTES:
            raise TransferFault('SOURCE_EXTRACT_SIZE_BOUND')
        for member in members:
            relative=Path(member.filename)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in member.filename
                    or stat.S_ISLNK(member.external_attr>>16)):
                raise TransferFault('SOURCE_UNSAFE_ARCHIVE_ENTRY')
            listed=expected[member.filename]
            if listed is not None and member.file_size!=listed:raise TransferFault('SOURCE_ARCHIVE_SIZE_MISMATCH')
            resolved[member.filename]=member.file_size
        # The ZIP's uncompressed size is a bound, even when the API omits a size.
        for member in members:
            dest=root/member.filename;dest.parent.mkdir(parents=True,exist_ok=True)
            count=0;digest=hashlib.sha256();limit=resolved[member.filename]
            with z.open(member) as source,dest.open('xb') as output:
                while True:
                    chunk=source.read(1024**2)
                    if not chunk:break
                    count+=len(chunk)
                    if count>limit:raise TransferFault('SOURCE_EXTRACT_SIZE_MISMATCH')
                    if disk_snapshot()['free_gib']<MIN_LOCAL_FREE_GIB:raise TransferFault('SOURCE_DISK_GUARD')
                    output.write(chunk);digest.update(chunk)
            if count!=limit:raise TransferFault('SOURCE_EXTRACT_SIZE_MISMATCH')
            hashes[member.filename]=digest.hexdigest()
    return {'sizes':resolved,'sha256':hashes}


def validate_source_root(root):
    root=Path(root)
    return (root/'src').is_dir() and all((root/name).is_file() for name in ('deploy_target.pkl','benchmark_initial.pkl','setup_commands.json'))


def download_source_bundle(session,report=None):
    report={} if report is None else report
    state={'handle':'dfranzen/taaf-kaggle-source-bundle-copy','version':1,'status':'LISTING'}
    report['source_bundle_transfer']=state
    persist=(lambda:save_report(report)) if report.get('test') else (lambda:None)
    persist();expected=source_inventory(session,state,persist)
    if disk_snapshot()['free_gib']<MIN_LOCAL_FREE_GIB+2*(MAX_SOURCE_BYTES+16*1024**2)/1024**3:
        raise TransferFault('SOURCE_INSUFFICIENT_DISK_HEADROOM')
    parent=Path('/content/arc3-preflight');parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='source-v1-',dir=parent))
    archive=staging/'source.zip';tree=staging/'content';tree.mkdir()
    try:
        print(f'Source bundle v1: {len(expected)} files; downloading the ZIP.',flush=True)
        state['status']='DOWNLOADING';persist()
        with requests.Session() as public:
            public.trust_env=False
            with source_archive_stream(session,public,state,persist) as chunks:
                total=0;start=time.monotonic();progress=start
                with archive.open('xb') as output:
                    for chunk in chunks:
                        total+=len(chunk)
                        if total>MAX_SOURCE_BYTES+16*1024**2:raise TransferFault('SOURCE_ARCHIVE_SIZE_BOUND')
                        if time.monotonic()-start>300:raise TransferFault('SOURCE_DOWNLOAD_TIME_BOUND')
                        if disk_snapshot()['free_gib']<MIN_LOCAL_FREE_GIB:raise TransferFault('SOURCE_DISK_GUARD')
                        output.write(chunk)
                        if time.monotonic()-progress>=30:
                            print(f'Source bundle: {total/1024**2:.1f} MiB read.',flush=True);progress=time.monotonic()
        verified=extract_source_zip(archive,expected,tree);archive.unlink()
        if not validate_source_root(tree):raise TransferFault('SOURCE_BUNDLE_LAYOUT_INVALID')
        state.update(status='DOWNLOADED_VERSION_1',path=str(tree),bytes=sum(verified['sizes'].values()),
            inventory=verified['sizes'],sha256=verified['sha256'],
            integrity_note='Exact pinned file names, listed known sizes, ZIP sizes/CRC and local SHA256; omitted API sizes resolved from ZIP.')
        persist();print('Source bundle v1: file list, sizes and ZIP CRC verified.',flush=True)
        return dict(state)
    except BaseException as error:
        shutil.rmtree(staging)
        state.update(status='FAILED',error_type=type(error).__name__,error_code=getattr(error,'code','SEE_ERROR_TYPE'),
            http_status=getattr(error,'http_status',None),partial_download_removed=True)
        persist();raise

import signal

class SetupFault(RuntimeError):

    def __init__(self, code):
        super().__init__(code)
        self.code = code

def save_report(report):
    path = Path(REPORT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

def stop_owned_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)

def guarded_command(args, env, log_path):
    if disk_snapshot()['free_gib'] < MIN_LOCAL_FREE_GIB:
        return {'status': 'STOPPED_LOCAL_DISK_GUARD', 'returncode': None}
    started = time.monotonic()
    progress = started
    minimum = disk_snapshot()['free_gib']
    reason = None
    print('Running:', Path(args[0]).name, ' '.join(args[1:4]), flush=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('wb') as log:
        process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                free = disk_snapshot()['free_gib']
                minimum = min(minimum, free)
                if free < MIN_LOCAL_FREE_GIB:
                    reason = 'STOPPED_LOCAL_DISK_GUARD'
                    stop_owned_process(process)
                    break
                if time.monotonic() - started > COMMAND_TIMEOUT_SECONDS:
                    reason = 'STOPPED_COMMAND_TIMEOUT'
                    stop_owned_process(process)
                    break
                if time.monotonic() - progress > 30:
                    print(f'  {int(time.monotonic() - started)} s; local free disk {free:.1f} GiB', flush=True)
                    progress = time.monotonic()
                time.sleep(1)
        finally:
            stop_owned_process(process)
    with log_path.open('rb') as log:
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 20000))
        tail = log.read().decode(errors='replace')
    result = {'status': reason or ('COMMAND_OK' if process.returncode == 0 else 'COMMAND_FAILED'), 'returncode': process.returncode, 'seconds': round(time.monotonic() - started, 3), 'observed_min_free_gib': minimum, 'log_path': str(log_path), 'log_tail': tail}
    if result['status'] != 'COMMAND_OK':
        print(tail[-4000:])
    return result

def bootstrap_uv(wheels, prefix):
    candidates = sorted(Path(wheels).glob('uv-*.whl'))
    if len(candidates) != 1:
        raise RuntimeError('Exactly one bundled uv wheel is required')
    output = Path(prefix) / 'bin/uv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(candidates[0]) as archive:
        members = [m for m in archive.infolist() if Path(m.filename).name == 'uv' and (not m.is_dir())]
        if len(members) != 1 or members[0].file_size > 128 * 1024 ** 2:
            raise RuntimeError('Bounded bundled uv executable not found')
        binary = archive.read(members[0])
    if not binary.startswith(b'\x7fELF'):
        raise RuntimeError('Bundled uv is not an ELF executable')
    output.write_bytes(binary)
    output.chmod(493)
    return str(output)

def discover_cuda(venv):
    roots = sorted({p.resolve() for p in Path(venv).glob('lib/python*/site-packages/nvidia')})
    nvcc = sorted({p.resolve() for root in roots for p in root.glob('**/bin/nvcc') if p.is_file()})
    runtimes = sorted({p.resolve() for root in roots for p in root.glob('**/libcudart.so*') if p.is_file() and 'stubs' not in p.parts})
    libraries = sorted({p.resolve() for root in roots for pattern in ('**/lib', '**/lib64') for p in root.glob(pattern) if p.is_dir() and 'stubs' not in p.parts})
    return {'nvidia_package_roots': [str(p) for p in roots], 'nvcc_candidates': [str(p) for p in nvcc], 'cudart_candidates': [str(p) for p in runtimes], 'library_directories': [str(p) for p in libraries]}

def discover_driver(env):
    directories = ['/usr/local/nvidia/lib64', '/usr/local/nvidia/lib', '/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/usr/lib64-nvidia', '/usr/local/cuda/compat']
    directories += [p for p in env.get('LD_LIBRARY_PATH', '').split(':') if p]
    candidates = {p.resolve() for directory in directories for p in Path(directory).glob('libcuda.so*') if p.is_file() and 'stubs' not in p.parts}
    ldconfig = shutil.which('ldconfig') or ('/sbin/ldconfig' if Path('/sbin/ldconfig').is_file() else None)
    if ldconfig:
        try:
            result = subprocess.run([ldconfig, '-p'], env=env, capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines():
                if re.match('\\s*libcuda\\.so', line) and ' => ' in line:
                    path = Path(line.split(' => ', 1)[1].strip())
                    if path.is_file() and 'stubs' not in path.parts:
                        candidates.add(path.resolve())
        except (OSError, subprocess.TimeoutExpired):
            pass
    return sorted(candidates)
DRIVER_PROBE_SCRIPT = "\nimport ctypes,json,pathlib\nctypes.CDLL('libcuda.so.1')\npaths=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libcuda.so' in line and line.split()[-1].startswith('/')})\nprint(json.dumps({'loaded_driver_paths':paths}))\n"

def runtime_environment(prefix, python, report=None):
    venv = Path(python).parent.parent
    discovery = discover_cuda(venv)
    if report is not None:
        report['cuda_discovery'] = discovery
        save_report(report)
    if not discovery['nvcc_candidates']:
        raise SetupFault('BUNDLED_NVCC_NOT_FOUND')
    if len(discovery['nvcc_candidates']) != 1:
        raise SetupFault('MULTIPLE_BUNDLED_NVCC_CANDIDATES')
    if not discovery['cudart_candidates']:
        raise SetupFault('BUNDLED_CUDART_NOT_FOUND')
    cuda = Path(discovery['nvcc_candidates'][0]).parent.parent
    lib = cuda / 'lib'
    lib.mkdir(exist_ok=True)
    lib64 = cuda / 'lib64'
    if not lib64.exists() and (not lib64.is_symlink()):
        lib64.symlink_to('lib')
    for source in sorted(lib.glob('*.so.*')):
        dest = lib / re.sub('\\.so\\..*$', '.so', source.name)
        if not dest.exists() and (not dest.is_symlink()):
            dest.symlink_to(source.name)
    cudart = lib / 'libcudart.so'
    if not cudart.exists():
        if cudart.is_symlink():
            cudart.unlink()
        cudart.symlink_to(Path(discovery['cudart_candidates'][0]))
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
        env.pop(name, None)
    env['PYTHONNOUSERSITE'] = '1'
    library_dirs = list(dict.fromkeys([str(lib64), str(lib)] + discovery['library_directories']))
    env['LD_LIBRARY_PATH'] = ':'.join(library_dirs + [env.get('LD_LIBRARY_PATH', '')])
    driver = discover_driver(env)
    if not driver:
        probe = guarded_command([python, '-c', DRIVER_PROBE_SCRIPT], env, Path(prefix) / 'logs/driver-discovery.log')
        if report is not None:
            report['driver_probe'] = probe
            save_report(report)
        if probe['status'] == 'COMMAND_OK':
            try:
                data = json.loads(probe['log_tail'].splitlines()[-1])
                driver = [Path(p) for p in data['loaded_driver_paths'] if Path(p).is_file() and 'stubs' not in Path(p).parts]
            except (ValueError, KeyError, TypeError):
                pass
    if report is not None:
        report['cuda_discovery']['driver_candidates'] = [str(p) for p in driver]
        save_report(report)
    link = lib / 'libcuda.so'
    if not link.exists():
        if not driver:
            raise SetupFault('GPU_DRIVER_LIBRARY_NOT_FOUND')
        if link.is_symlink():
            link.unlink()
        link.symlink_to(driver[0].resolve())
    if driver:
        env['LD_LIBRARY_PATH'] = ':'.join(library_dirs + [str(driver[0].parent), os.environ.get('LD_LIBRARY_PATH', '')])
    cxx = next((c for c in ('g++-15', 'g++-14', 'g++-13', 'g++-12', 'g++-11', 'g++') if shutil.which(c)), None)
    if not cxx or not shutil.which(cxx.replace('g++', 'gcc')):
        raise SetupFault('MATCHING_HOST_COMPILERS_NOT_FOUND')
    cache = str(Path(prefix) / 'cache')
    env.update({'PYTHONNOUSERSITE': '1', 'UV_OFFLINE': '1', 'UV_PYTHON_DOWNLOADS': 'never', 'UV_NO_CACHE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'CUDA_HOME': str(cuda), 'CUDACXX': str(cuda / 'bin/nvcc'), 'PATH': str(cuda / 'bin') + ':' + str(Path(python).parent) + ':' + env.get('PATH', ''), 'CC': cxx.replace('g++', 'gcc'), 'CXX': cxx, 'CUDAHOSTCXX': cxx, 'TORCH_CUDA_ARCH_LIST': '12.0', 'XDG_CACHE_HOME': cache, 'HF_HOME': cache + '/huggingface', 'TORCH_HOME': cache + '/torch', 'CUDA_CACHE_PATH': cache + '/cuda', 'TRITON_CACHE_DIR': cache + '/triton', 'FLASHINFER_WORKSPACE_BASE': cache + '/flashinfer', 'SGLANG_CACHE_DIR': cache + '/sglang', 'SGLANG_JIT_CACHE_DIR': cache + '/sglang/jit', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'TOKENIZERS_PARALLELISM': 'false'})
    return env
CHECK_SCRIPT = "\nimport sys,json,importlib.metadata as m\nimport torch,sglang\nversions={name:m.version(name) for name in ('torch','sglang','transformers','safetensors','flashinfer-python')}\nassert sys.version_info[:2] == (3,12), 'Wrong Python ABI'\nassert versions['torch']=='2.13.0+cu130', 'Unexpected torch version'\nassert versions['sglang']=='0.5.19+gd00d88efc8d6', 'Unexpected SGLang version'\nassert torch.version.cuda and torch.version.cuda.startswith('13.'), 'Wrong CUDA build'\nassert torch.cuda.is_available(), 'CUDA unavailable'\nassert torch.cuda.get_device_capability(0)==(12,0), 'Unexpected GPU architecture'\nx=torch.tensor([1.,4.,9.], device='cuda'); y=x.sqrt(); torch.cuda.synchronize()\nassert y.cpu().tolist()==[1.,2.,3.]\nprint(json.dumps({'python':sys.version.split()[0],'versions':versions,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'small_gpu_operation':'PASSED'}))\n"

def loader_sources(venv):
    candidates = list(Path(venv).glob('lib/python*/site-packages/sglang/srt'))
    if len(candidates) != 1:
        return {'status': 'SGLANG_SOURCE_NOT_FOUND'}
    root = candidates[0]
    terms = ('ple_offload_embedding', 'prefetch_checkpoints', 'drop_cache_after_load', 'safetensors_weights_iterator', 'mmap', 'madvise', 'posix_fadvise')
    files = sorted(root.rglob('*.py'), key=lambda p: (0 if 'qwen4' in p.name else 1 if 'weight' in p.name else 2, str(p)))
    excerpts = []
    for path in files:
        if path.stat().st_size > 2 * 1024 ** 2:
            continue
        text = path.read_text(errors='replace')
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if any((term in line for term in terms))]
        if not hits:
            continue
        indices = sorted({j for i in hits for j in range(max(0, i - 20), min(len(lines), i + 45))})
        excerpts.append({'file': str(path.relative_to(root)), 'source_sha256': hashlib.sha256(text.encode()).hexdigest(), 'excerpt': '\n'.join((f'{i + 1}: {lines[i]}' for i in indices[:650])), 'truncated': len(indices) > 650, 'terms_found': [term for term in terms if term in text]})
        if len(excerpts) >= 16:
            break
    return {'status': 'COLLECTED', 'excerpts': excerpts, 'note': 'Read-only excerpts of the installed pinned loader. No loader code was patched and no model weights were read.'}

def prepare_python312(report, wheel_root):
    prefix = Path(SETUP_ROOT)
    wheels = Path(wheel_root) / 'wheels'
    lock = Path(wheel_root) / 'requirements.lock'
    if not wheels.is_dir() or not lock.is_file():
        raise RuntimeError('Wheelhouse layout invalid')
    owner = prefix / 'arc3-setup-owner.json'
    identity_name = {'workflow': 'ARC3 pinned Python312 setup', 'version': 1}
    if prefix.exists() and any(prefix.iterdir()) and (not owner.is_file() or json.loads(owner.read_text()) != identity_name):
        report['installation'] = {'status': 'UNRECOGNIZED_EXISTING_SETUP_FOLDER'}
        return
    prefix.mkdir(parents=True, exist_ok=True)
    if not owner.exists():
        owner.write_text(json.dumps(identity_name))
    report['python312'] = guarded_command([PYTHON312, '-c', 'import sys,json; assert sys.version_info[:2]==(3,12); print(json.dumps({"python":sys.version.split()[0],"executable":sys.executable}))'], os.environ.copy(), prefix / 'logs/python312.log')
    save_report(report)
    if report['python312']['status'] != 'COMMAND_OK':
        return
    if disk_snapshot()['free_gib'] < MIN_INSTALL_START_FREE_GIB:
        report['installation'] = {'status': 'INSUFFICIENT_INSTALL_HEADROOM'}
        return
    uv = bootstrap_uv(wheels, prefix)
    env = dict(os.environ, UV_OFFLINE='1', UV_PYTHON_DOWNLOADS='never', UV_NO_CACHE='1', PYTHONNOUSERSITE='1')
    for name in ('PYTHONPATH', 'KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
        env.pop(name, None)
    venv = prefix / 'venv'
    python = str(venv / 'bin/python')
    report['venv_path'] = str(venv)
    report['commands'] = []
    identity = hashlib.sha256(lock.read_bytes() + '\n'.join(sorted((p.name for p in wheels.glob('*.whl')))).encode()).hexdigest()
    marker = prefix / 'installed-bundle.json'
    if (venv / 'bin/python').exists():
        checked = guarded_command([python, '-c', 'import sys; assert sys.version_info[:2]==(3,12)'], env, prefix / 'logs/existing-python.log')
        if checked['status'] != 'COMMAND_OK':
            report['installation'] = {'status': 'EXISTING_VENV_WRONG_PYTHON'}
            return
    else:
        result = guarded_command([uv, 'venv', '--python', PYTHON312, str(venv)], env, prefix / 'logs/venv.log')
        report['commands'].append(dict(step='create_venv', **result))
        save_report(report)
        if result['status'] != 'COMMAND_OK':
            return
    matching = marker.is_file() and json.loads(marker.read_text()).get('bundle_identity') == identity
    if not matching:
        install = [uv, 'pip', 'install', '--python', python, '--no-index', '--find-links', str(wheels)]
        result = guarded_command(install + ['-r', str(lock)], env, prefix / 'logs/install.log')
        report['commands'].append(dict(step='install_locked_wheels', **result))
        save_report(report)
        if result['status'] != 'COMMAND_OK':
            return
        sglang_wheels = list(wheels.glob('sglang-*.whl'))
        if len(sglang_wheels) != 1:
            raise RuntimeError('Expected exactly one custom SGLang wheel')
        result = guarded_command(install + ['--reinstall', '--no-deps', str(sglang_wheels[0])], env, prefix / 'logs/sglang.log')
        report['commands'].append(dict(step='custom_sglang_wheel', **result))
        save_report(report)
        if result['status'] != 'COMMAND_OK':
            return
        marker.write_text(json.dumps({'bundle_identity': identity, 'wheelhouse_version': 1, 'python_major_minor': '3.12'}))
    report['installation'] = {'status': 'INSTALLED_OR_REUSED', 'bundle_identity': identity}
    report['dependency_check'] = guarded_command([uv, 'pip', 'check', '--python', python], env, prefix / 'logs/dependencies.log')
    save_report(report)
    report['stage'] = 'collect_loader_sources'
    report['loader_sources'] = loader_sources(venv)
    report['loader_sources']['model_weight_prefetch_started'] = False
    save_report(report)
    report['stage'] = 'prepare_cuda_paths'
    save_report(report)
    env = runtime_environment(prefix, python, report)
    report['stage'] = 'check_cuda_toolkit'
    save_report(report)
    report['cuda_toolkit'] = guarded_command([env['CUDACXX'], '--version'], env, prefix / 'logs/cuda.log')
    report['stage'] = 'check_gpu_runtime'
    save_report(report)
    report['runtime_check'] = guarded_command([python, '-c', CHECK_SCRIPT], env, prefix / 'logs/runtime.log')
    names = ('PYTHONNOUSERSITE', 'UV_OFFLINE', 'UV_PYTHON_DOWNLOADS', 'UV_NO_CACHE', 'CUDA_HOME', 'CUDACXX', 'CC', 'CXX', 'CUDAHOSTCXX', 'TORCH_CUDA_ARCH_LIST')
    report['runtime_configuration'] = {name: env[name] for name in names}
    report['python_executable'] = python

def run_setup():
    report = {'test': 'ARC3 pinned Python 3.12 setup', 'version': 2, 'status': 'RUNNING', 'utc': datetime.now(timezone.utc).isoformat(), 'environment': environment_snapshot(), 'auxiliary_resources': [], 'model_weights_read': False, 'main_model_loaded': False, 'model_weight_download_started': False, 'drive_quota_checked': False, 'inference_settings_changed': False, 'notebook_kernel_changed': False, 'dataset_versions': {'wheelhouse': 1, 'source_bundle': 1}}
    save_report(report)
    session = None
    try:
        session, mode = authenticated_session()
        report['auth_mode_used'] = mode
        for resource in RESOURCES:
            print('Mounting:', resource['name'], flush=True)
            record = mount_auxiliary_resource(session, resource)
            if resource['name'] == 'source_bundle' and record.get('mount', {}).get('status') != 'MOUNTED':
                for attempt in range(2):
                    time.sleep(3)
                    record = mount_auxiliary_resource(session, resource)
                    if record.get('mount', {}).get('status') == 'MOUNTED':
                        break
                if record.get('mount', {}).get('status') != 'MOUNTED' and ALLOW_BOUNDED_SOURCE_DOWNLOAD:
                    print('Trying a bounded download of source bundle version 1 (at most 512 MiB).', flush=True)
                    try:
                        record['bounded_download'] = download_source_bundle(session)
                    except Exception as error:
                        record['bounded_download'] = {'status': getattr(error, 'code', 'SOURCE_DOWNLOAD_ERROR'), 'http_status': getattr(error, 'http_status', None), 'error_type': type(error).__name__}
            report['auxiliary_resources'].append(record)
            save_report(report)
            print(' ->', record.get('bounded_download', record.get('mount', record['support']))['status'], flush=True)
    except Exception as error:
        report['kaggle_error_type'] = type(error).__name__
    finally:
        if session is not None:
            session.headers.pop('Authorization', None)
            session.auth = None
            session.close()
    wheel = next((r for r in report['auxiliary_resources'] if r['name'] == 'wheelhouse'), {})
    try:
        if wheel.get('mount', {}).get('status') == 'MOUNTED':
            prepare_python312(report, wheel['mount']['path'])
        else:
            report['installation'] = {'status': 'WHEELHOUSE_NOT_MOUNTED'}
    except Exception as error:
        report['setup_error_type'] = type(error).__name__
        report['setup_error_code'] = getattr(error, 'code', 'UNEXPECTED_SETUP_ERROR')
        print('Setup error:', report['setup_error_code'], '| stage:', report.get('stage', 'installation'), flush=True)
    report['environment_after'] = environment_snapshot()
    source = next((r for r in report['auxiliary_resources'] if r['name'] == 'source_bundle'), {})
    source_ok = source.get('mount', {}).get('status') == 'MOUNTED' or source.get('bounded_download', {}).get('status') == 'DOWNLOADED_VERSION_1'
    report['source_bundle_ready'] = source_ok
    checks_ok = all((report.get(name, {}).get('status') == 'COMMAND_OK' for name in ('runtime_check', 'dependency_check', 'cuda_toolkit')))
    report['status'] = 'ENVIRONMENT_READY' if checks_ok and source_ok else 'BLOCKERS_REMAIN'
    report['full_model_loading_ready'] = False
    save_report(report)
    print('Python 3.12 setup:', report.get('installation', {}).get('status', 'INCOMPLETE'))
    print('CUDA / small GPU test:', report.get('runtime_check', {}).get('status', 'NOT_RUN'))
    print('Source bundle ready:', source_ok)
    print('Result:', report['status'])
    print('Report:', REPORT_PATH)
    return report

import copy
import shlex
import socket
import traceback

# Original draft metadata rules; required by prepare_draft_view.
EXPERT_TARGET = r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"

SERVED_MODEL_NAME = "flashnext"
CFG = dict(
    CTX=(116+12+8)*1024,
    MEMFRAC=0.96,
    MAXREQ=10,
    CUDAGRAPH_MAXBS=10,
    MAMBA_CACHE=60,
    CHUNK=8192,
    MAX_PREFILL=16384,
    SPEC=True,
    FRSPEC=True,
    AUTOTUNE=True,
    # Preserve the original Kaggle checkpoint prefetch/cache behavior.
    PREFETCH_CHECKPOINTS=True,
    DROP_CACHE_AFTER_LOAD=True,
    LINEAR_BACKEND="flashinfer",
    KVDTYPE="fp8_e4m3",
    SSM_DTYPE="bfloat16",
    MAMBA_RADIX="extra_buffer",
    GDN_MTP_CACHE_MODE="none",
    SERVED_NAME=SERVED_MODEL_NAME,
    SPEC_ACCEPT_SINGLE=1.0,
    SPEC_ACCEPT_ACC=1.0,
    SPEC_STEPS=3,
)
TOKEN_MAP_SHA = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"

def find_unique(root, name, *, directory=False, required=True):
    root = Path(root)
    direct = root if root.name == name else root / name
    valid = lambda p: p.is_dir() if directory else p.is_file()
    if valid(direct): return direct
    hits = sorted(p for p in root.rglob(name) if valid(p))
    if len(hits) > 1:
        raise RuntimeError(f"Multiple {name} matches under {root}; use a more specific directory: {hits}")
    if hits: return hits[0]
    if required: raise FileNotFoundError(f"{name} not found under {root}")
    return None

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()

def indexed_shards(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names: raise RuntimeError(f"Empty weight index: {root}")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Invalid shard path: {name}")
        p = root / rel
        if not p.is_file() or p.stat().st_size < 8:
            raise RuntimeError(f"Missing/empty shard: {p}")
    return index, names

def prepare_draft_view(target, source_root, work_root):
    """Adapt metadata only; link original dataset weights without copying them."""
    target, source_root, work_root = map(Path, (target, source_root, work_root))
    configs = [source_root / "config.json"] if (source_root / "config.json").is_file() else []
    configs += sorted(p for p in source_root.rglob("config.json") if p not in configs)
    candidates = []
    for path in configs:
        cfg = json.loads(path.read_text())
        q = cfg.get("quantization_config", {})
        if q.get("quant_method") == "compressed-tensors" and "mtp_routed_experts" in q.get("config_groups", {}):
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one INT4 g32 MTP checkpoint under DRAFT_MODEL_DIR={source_root}; found {len(candidates)}")
    source, cfg = candidates[0]
    index, shards = indexed_shards(source)
    if not any(k.startswith("mtp.layers.0.mlp.experts.") for k in index["weight_map"]):
        raise RuntimeError("Draft index does not contain the expected MTP experts")
    cfg = copy.deepcopy(cfg)
    q = cfg["quantization_config"]
    group = q["config_groups"]["mtp_routed_experts"]
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["symmetric"]) != (4, 32, True):
        raise RuntimeError("Expected symmetric INT4 group32 draft experts")
    if group["targets"] not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise RuntimeError(f"Unexpected draft target rules: {group['targets']}")
    if q.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise RuntimeError("Unexpected draft ignore rules")
    group["targets"] = [EXPERT_TARGET]
    q["ignore"] = [DENSE_IGNORE]
    # Content/path-specific directory makes notebook reruns idempotent without
    # overwriting an unrelated view. We never modify the read-only datasets.
    identity = json.dumps({"source": str(source.resolve()), "target": str(target.resolve()), "config": cfg}, sort_keys=True)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(cfg, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise RuntimeError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (target / name).is_file(): links[name] = target / name
    for name, src in links.items():
        dest = view / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_symlink() or dest.resolve() != src.resolve():
                raise RuntimeError(f"Unexpected existing draft-view file: {dest}")
        else:
            dest.symlink_to(src.resolve())
    print(f"Draft source: {source}\nSGLang draft view: {view} ({len(shards)} linked weight shards)")
    return view

CFG = {'CTX': 139264, 'MEMFRAC': 0.96, 'MAXREQ': 10, 'CUDAGRAPH_MAXBS': 10, 'MAMBA_CACHE': 60, 'CHUNK': 8192, 'MAX_PREFILL': 16384, 'SPEC': True, 'FRSPEC': True, 'AUTOTUNE': True, 'PREFETCH_CHECKPOINTS': True, 'DROP_CACHE_AFTER_LOAD': True, 'LINEAR_BACKEND': 'flashinfer', 'KVDTYPE': 'fp8_e4m3', 'SSM_DTYPE': 'bfloat16', 'MAMBA_RADIX': 'extra_buffer', 'GDN_MTP_CACHE_MODE': 'none', 'SERVED_NAME': 'flashnext', 'SPEC_ACCEPT_SINGLE': 1.0, 'SPEC_ACCEPT_ACC': 1.0, 'SPEC_STEPS': 3}

STAGED_LOADER_SOURCE = """\"\"\"Scoped transport hook: original safetensors decoding and SGLang weight loading.\"\"\"
import hashlib
import hmac
import concurrent.futures
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.request

STUB_BYTES = b'ARC3STUB'

class StageStore:
    def __init__(self, root, manifest, proxy, token, log, free_floor=15, connections=8, shard_timeout=600):
        self.root = Path(root).resolve()
        self.items = {item['name']: item for item in manifest if item['name'].endswith('.safetensors')}
        self.proxy, self.token, self.log = proxy.rstrip('/'), token, Path(log)
        self.free_floor = free_floor
        if not 1<=connections<=8 or shard_timeout<=0:raise RuntimeError('Invalid staged transport bounds')
        self.connections,self.shard_timeout=connections,shard_timeout
        self.lock = threading.RLock()
        self.leases = {}
    def match(self, path):
        try: relative = Path(path).resolve().relative_to(self.root).as_posix()
        except (ValueError, TypeError): return None
        return relative if relative in self.items else None
    def record(self, event, **fields):
        with self.log.open('a') as output:
            output.write(json.dumps(dict(event=event, utc_seconds=time.time(), **fields))+'\\n')
    def acquire(self, name):
        # One staged shard at a time; its byte ranges download concurrently.
        with self.lock:
            path = self.root / name
            if self.leases.get(name, 0):
                self.leases[name] += 1
                return path
            item = self.items[name]
            if path.is_symlink() or path.stat().st_size != len(STUB_BYTES) or path.read_bytes() != STUB_BYTES:
                raise RuntimeError('Unexpected non-placeholder staging file')
            free = shutil.disk_usage(self.root).free / 1024**3
            if free < self.free_floor + item['bytes']/1024**3:
                raise RuntimeError('Insufficient disk headroom for one shard')
            temporary = path.with_name(path.name+'.part')
            start = time.monotonic();stop=threading.Event();digest=hashlib.sha256()
            connections=min(self.connections,item['bytes']);deadline=start+self.shard_timeout
            self.record('stage_start', name=name, expected_bytes=item['bytes'], connections=connections)
            try:
                with temporary.open('xb') as output:output.truncate(item['bytes'])
                def download_range(number):
                    import http.client
                    import urllib.error
                    first=number*item['bytes']//connections
                    last=(number+1)*item['bytes']//connections-1
                    expected=last-first+1;count=0
                    try:
                        for attempt in range(1,5):
                            if stop.is_set():raise RuntimeError('Staging cancelled')
                            if time.monotonic()>deadline:raise RuntimeError('Shard staging timeout')
                            offset=first+count
                            request=urllib.request.Request(self.proxy+'/weight/'+name,
                                headers={'Authorization':'Bearer '+self.token,'Range':f'bytes={offset}-{last}'})
                            window_start=time.monotonic();window_count=count
                            try:
                                with urllib.request.urlopen(request,timeout=min(20,max(0.1,deadline-time.monotonic()))) as source, temporary.open('r+b',buffering=0) as output:
                                    if source.status!=206 or source.headers.get('Content-Range')!=f'bytes {offset}-{last}/{item["bytes"]}':
                                        raise RuntimeError('Proxy content range mismatch')
                                    if int(source.headers.get('Content-Length',-1))!=expected-count:
                                        raise RuntimeError('Proxy content length mismatch')
                                    output.seek(offset)
                                    while count<expected:
                                        if stop.is_set():raise RuntimeError('Staging cancelled')
                                        if time.monotonic()>deadline:raise RuntimeError('Shard staging timeout')
                                        data=source.read1(min(256*1024,expected-count))
                                        if not data:raise ConnectionError('Incomplete staged range')
                                        if shutil.disk_usage(self.root).free/1024**3<self.free_floor:
                                            raise RuntimeError('Local disk guard during staging')
                                        view=memoryview(data)
                                        while view:
                                            written=output.write(view)
                                            if not written:raise RuntimeError('Staging write failed')
                                            count+=written;view=view[written:]
                                        now=time.monotonic();elapsed=now-window_start
                                        if elapsed>=30:
                                            speed=(count-window_count)/elapsed
                                            self.record('range_progress',name=name,connection=number+1,
                                                attempt=attempt,bytes=count,expected_bytes=expected,
                                                mb_per_second=round(speed/1e6,2))
                                            if count<expected and speed<1024**2:
                                                raise ConnectionError('Range transfer too slow')
                                            window_start=now;window_count=count
                                self.record('range_complete',name=name,connection=number+1,attempt=attempt,bytes=count)
                                return count
                            except (OSError,http.client.HTTPException) as error:
                                if isinstance(error,urllib.error.HTTPError) and error.code not in (408,429,500,502,503,504):raise
                                if getattr(error,'errno',None) in (28,13):raise
                                if stop.is_set():raise RuntimeError('Staging cancelled') from None
                                if attempt==4 or time.monotonic()>deadline:raise
                                self.record('range_retry',name=name,connection=number+1,attempt=attempt,
                                    bytes=count,expected_bytes=expected,error_type=type(error).__name__)
                                stop.wait(min(2**(attempt-1),max(0,deadline-time.monotonic())))
                    except BaseException:
                        stop.set();raise
                pool=concurrent.futures.ThreadPoolExecutor(max_workers=connections)
                try:
                    futures=[pool.submit(download_range,i) for i in range(connections)]
                    count=sum(f.result() for f in concurrent.futures.as_completed(futures))
                finally:
                    stop.set();pool.shutdown(wait=True,cancel_futures=True)
                download_seconds=time.monotonic()-start
                hash_start=time.monotonic()
                with temporary.open('rb') as source:
                    for data in iter(lambda:source.read(8*1024**2),b''):digest.update(data)
                checksum_seconds=time.monotonic()-hash_start
                if count != item['bytes'] or not hmac.compare_digest(digest.hexdigest(), item['sha256']):
                    raise RuntimeError('Shard size or SHA256 mismatch')
                temporary.replace(path)
                self.leases[name] = 1
                seconds=time.monotonic()-start
                self.record('stage_verified', name=name, bytes=count, seconds=round(seconds,3),
                            connections=connections, download_seconds=round(download_seconds,3),
                            checksum_seconds=round(checksum_seconds,3),
                            transfer_mb_per_second=round(count/download_seconds/1e6,2),
                            verified_mb_per_second=round(count/seconds/1e6,2))
                return path
            except BaseException:
                temporary.unlink(missing_ok=True)
                self.record('stage_failed', name=name)
                raise
    def release(self, name):
        with self.lock:
            count = self.leases.get(name, 0)
            if count < 1: raise RuntimeError('Invalid shard lease release')
            if count > 1:
                self.leases[name] = count-1
                return
            del self.leases[name]
            path = self.root/name
            path.unlink()
            with path.open('xb') as output: output.write(STUB_BYTES)
            self.record('stage_removed', name=name)


class StagedSafeOpen:
    def __init__(self, native, store, name, args, kwargs):
        self.store, self.name = store, name
        store.acquire(name)
        self.closed = False
        try: self.context = native(*args, **kwargs)
        except BaseException:
            store.release(name); self.closed = True; raise
    def __enter__(self):
        try: self.context.__enter__(); return self
        except BaseException:
            self.close(); raise
    def __getattr__(self, name): return getattr(self.context, name)
    def __exit__(self, kind, value, traceback):
        try: return self.context.__exit__(kind, value, traceback)
        finally: self.close()
    def close(self):
        if self.closed: return
        self.closed = True
        # Finish H2D reads before unlinking the owned temporary backing file.
        torch = sys.modules.get('torch')
        try:
            if torch is not None and torch.cuda.is_initialized(): torch.cuda.synchronize()
        finally: self.store.release(self.name)


def patch_safetensors(module, store):
    native = module.safe_open
    def safe_open(*args, **kwargs):
        filename = args[0] if args else kwargs.get('filename')
        name = store.match(filename)
        if name is None: return native(*args, **kwargs)
        return StagedSafeOpen(native, store, name, args, kwargs)
    module.safe_open = safe_open


def patch_weight_iterator(module, store):
    native = module.safetensors_weights_iterator
    signature = inspect.signature(native)
    def iterator(*args, **kwargs):
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        files = bound.arguments['hf_weights_files']
        # Preserve file order, tensor order, dtype, values and the original loader.
        for path in files:
            name = store.match(path)
            if name is None:
                one = dict(bound.arguments, hf_weights_files=[path])
                yield from native(**one)
                continue
            store.acquire(name)
            try:
                one = dict(bound.arguments, hf_weights_files=[path])
                yield from native(**one)
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized(): torch.cuda.synchronize()
            finally: store.release(name)
    module.safetensors_weights_iterator = iterator
    store.record('iterator_hook_installed', module=module.__name__)


class HookLoader:
    def __init__(self, loader, patch): self.loader, self.patch = loader, patch
    def create_module(self, spec): return self.loader.create_module(spec)
    def exec_module(self, module):
        self.loader.exec_module(module); self.patch(module)
    def __getattr__(self, name): return getattr(self.loader, name)


class HookFinder(importlib.abc.MetaPathFinder):
    def __init__(self, patches): self.patches = patches
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.patches: return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = HookLoader(spec.loader, self.patches[fullname])
        return spec


def install_from_environment():
    if os.environ.get('ARC3_STAGED_LOADER') != '1': return
    root = os.environ['ARC3_STAGED_MODEL_DIR']
    manifest = json.loads(Path(os.environ['ARC3_STAGED_MANIFEST']).read_text())
    store = StageStore(root, manifest, os.environ['ARC3_STAGED_PROXY'], os.environ['ARC3_STAGED_TOKEN'],
                       os.environ['ARC3_STAGED_LOG'], float(os.environ.get('ARC3_STAGED_FREE_FLOOR', '15')),
                       int(os.environ.get('ARC3_STAGED_CONNECTIONS','8')),
                       float(os.environ.get('ARC3_STAGED_SHARD_TIMEOUT','600')))
    patches = {'safetensors': lambda module: patch_safetensors(module, store),
               'sglang.srt.model_loader.weight_utils': lambda module: patch_weight_iterator(module, store)}
    for name, patch in patches.items():
        if name in sys.modules: patch(sys.modules[name])
    sys.meta_path.insert(0, HookFinder(patches))
    store.record('transport_hook_installed', pid=os.getpid(), connections=store.connections)
"""

ORIGINAL_RUNTIME_FLAGS = """env.update({
    "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
    "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", ""),
    "CC": CC, "CXX": CXX, "CUDAHOSTCXX": CXX,
    "TORCH_CUDA_ARCH_LIST": "12.0",
    "HF_HOME": f"{cache}/huggingface", "XDG_CACHE_HOME": cache,
    "TORCH_HOME": f"{cache}/torch", "CUDA_CACHE_PATH": f"{cache}/cuda",
    "TORCHINDUCTOR_CACHE_DIR": f"{cache}/torchinductor", "TRITON_CACHE_DIR": f"{cache}/triton",
    "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
    "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
    "MAX_JOBS": "8", "CMAKE_BUILD_PARALLEL_LEVEL": "8",
    "FLASHINFER_NINJA_JOBS": "8", "FLASHINFER_NVCC_THREADS": "2", "TORCHINDUCTOR_COMPILE_THREADS": "8",
    "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_SM120_ONLINE_MXFP8": "0",
    "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
    "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false",
    "SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",
})
"""

ORIGINAL_ARGUMENTS = """
graph_bs = sorted({1, 2, 4, 7, 8, 9, 10, CFG["MAXREQ"], CFG["CUDAGRAPH_MAXBS"]})
graph_bs = [n for n in graph_bs if n <= CFG["CUDAGRAPH_MAXBS"]]
args = [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
    "--model-loader-extra-config", '{"enable_multithread_load":false}',
    "--served-model-name", str(CFG["SERVED_NAME"]), "--host", "127.0.0.1",
    "--port", str(SERVED_MODEL_PORT), "--tensor-parallel-size", "1",
    "--dtype", "bfloat16", "--quantization", "auto-round",
    "--kv-cache-dtype", CFG["KVDTYPE"], "--mem-fraction-static", str(CFG["MEMFRAC"]),
    "--context-length", str(CFG["CTX"]), "--page-size", "64",
    "--max-running-requests", str(CFG["MAXREQ"]), "--chunked-prefill-size", str(CFG["CHUNK"]),
    "--max-prefill-tokens", str(CFG["MAX_PREFILL"]),
    "--cuda-graph-max-bs-decode", str(CFG["CUDAGRAPH_MAXBS"]),
    "--cuda-graph-bs-decode", *map(str, graph_bs),
    "--mamba-ssm-dtype", CFG["SSM_DTYPE"], "--max-mamba-cache-size", str(CFG["MAMBA_CACHE"]),
    "--mamba-radix-cache-strategy", CFG["MAMBA_RADIX"], "--mamba-track-interval", "64",
    "--mamba-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-decode-backend", CFG["LINEAR_BACKEND"],
    "--linear-attn-prefill-backend", CFG["LINEAR_BACKEND"],
    "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
    "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
    "--default-chat-template-kwargs", '{"preserve_thinking":true}',
    "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
    "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging"]
if CFG["PREFETCH_CHECKPOINTS"]: args += ["--weight-loader-prefetch-checkpoints"]
if CFG["DROP_CACHE_AFTER_LOAD"]: args += ["--weight-loader-drop-cache-after-load"]
tmpl = Path(MODEL_DIR, "chat_template.jinja")
if tmpl.is_file(): args += ["--chat-template", str(tmpl)]
if not CFG["AUTOTUNE"]: args += ["--disable-flashinfer-autotune"]
if CFG["GDN_MTP_CACHE_MODE"]: args += ["--gdn-mtp-cache-mode", CFG["GDN_MTP_CACHE_MODE"]]
if CFG["SPEC"]:
    args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(CFG["SPEC_STEPS"]),
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1),
        "--speculative-draft-model-path", str(DRAFT_VIEW),
        "--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",
        "--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],
        "--speculative-accept-threshold-single", str(CFG["SPEC_ACCEPT_SINGLE"]),
        "--speculative-accept-threshold-acc", str(CFG["SPEC_ACCEPT_ACC"])]
    if CFG["FRSPEC"]:
        tok = find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt", required=False)
        if tok is None: tok = find_unique(WHEELHOUSE_DIR, "flash-next-64k.pt")
        assert sha256(tok) == TOKEN_MAP_SHA, "Wrong FR-Spec map; use the map from the pinned Pennyroyal bundle"
        assert sha256(Path(MODEL_DIR, "tokenizer.json")) == TOKENIZER_SHA, "Tokenizer differs from the FR-Spec map; set CFG['FRSPEC']=False"
        args += ["--speculative-token-map", str(tok)]

"""

import hmac
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote


class DriveReader:
    def __init__(self, credentials, manifest):
        self.credentials = credentials
        self.items = {item['name']: item for item in manifest}
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
    def open_range(self, name, start, end):
        from google.auth.transport.requests import AuthorizedSession, Request
        item = self.items[name]
        if not 0 <= start <= end < item['bytes']:
            raise SetupFault('DRIVE_RANGE_OUT_OF_BOUNDS')
        # Refresh once before concurrent requests; the network requests themselves
        # must not hold the refresh lock or their header latency serializes them.
        with self.lock:
            expiry = self.credentials.expiry
            if expiry is not None and expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if not self.credentials.valid or (expiry is not None and expiry <= datetime.now(timezone.utc)+timedelta(minutes=10)):
                self.credentials.refresh(Request())
        session = AuthorizedSession(self.credentials); session.trust_env = False
        try:
            if self.cancelled.is_set(): raise SetupFault('DRIVE_TRANSFER_CANCELLED')
            response = session.get('https://www.googleapis.com/drive/v3/files/'+item['drive_file_id'],
                                   params={'alt':'media'}, headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'},
                                   stream=True, timeout=(15,60), allow_redirects=False)
            try:
                if response.status_code != 206:
                    raise SetupFault('DRIVE_RANGE_HTTP_'+str(response.status_code))
                if response.headers.get('Content-Range') != f'bytes {start}-{end}/{item["bytes"]}':
                    raise SetupFault('DRIVE_CONTENT_RANGE_MISMATCH')
                if response.headers.get('Content-Length') and int(response.headers['Content-Length']) != end-start+1:
                    raise SetupFault('DRIVE_RANGE_SIZE_MISMATCH')
                if response.headers.get('Content-Encoding','identity').lower() != 'identity':
                    raise SetupFault('DRIVE_RANGE_ENCODING_MISMATCH')
                return session, response
            except BaseException:
                response.close(); raise
        except BaseException:
            session.close(); raise
    def open(self, name):
        from google.auth.transport.requests import AuthorizedSession
        item = self.items[name]
        session = AuthorizedSession(self.credentials); session.trust_env = False
        try:
            with self.lock:
                response = session.get('https://www.googleapis.com/drive/v3/files/'+item['drive_file_id'],
                                       params={'alt':'media'}, stream=True, timeout=(15,120), allow_redirects=False)
            if response.status_code != 200:
                response.close(); raise SetupFault('DRIVE_MEDIA_HTTP_'+str(response.status_code))
            if response.headers.get('Content-Length') and int(response.headers['Content-Length']) != item['bytes']:
                response.close(); raise SetupFault('DRIVE_MEDIA_SIZE_MISMATCH')
            return session, response
        except BaseException:
            session.close(); raise
    def metadata(self, name, limit=64*1024**2):
        item = self.items[name]
        if item['bytes'] > limit or name.endswith('.safetensors'):
            raise RuntimeError('Metadata read bound exceeded')
        session, response = self.open(name)
        try:
            data = bytearray()
            for chunk in response.iter_content(1024**2):
                data.extend(chunk)
                if len(data) > item['bytes']: raise RuntimeError('Metadata exceeds expected size')
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise RuntimeError('Metadata checksum mismatch')
            return data
        finally:
            response.close(); session.close()


def start_drive_proxy(reader, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''), 'Bearer '+token):
                self.send_error(403); return
            if not self.path.startswith('/weight/'):
                self.send_error(404); return
            name = unquote(self.path[len('/weight/'):])
            if name not in reader.items or not name.endswith('.safetensors'):
                self.send_error(404); return
            byte_range=self.headers.get('Range')
            selected=None
            if byte_range is not None:
                match=re.fullmatch(r'bytes=(\d+)-(\d+)',byte_range) if len(byte_range)<=100 else None
                if match is None:
                    self.send_error(416);return
                start,end=map(int,match.groups())
                if not 0<=start<=end<reader.items[name]['bytes']:
                    self.send_error(416);return
                selected=(start,end)
            session = response = None
            sent_headers=False
            try:
                if selected is None:
                    session,response=reader.open(name)
                    expected=reader.items[name]['bytes']
                    self.send_response(200)
                else:
                    start,end=selected
                    session,response=reader.open_range(name,start,end)
                    expected=end-start+1
                    self.send_response(206)
                    self.send_header('Content-Range',f'bytes {start}-{end}/{reader.items[name]["bytes"]}')
                self.send_header('Content-Length', str(expected))
                self.send_header('Content-Type', 'application/octet-stream'); self.end_headers()
                sent_headers=True;count=0
                for chunk in response.iter_content(1024**2 if selected is not None else 8*1024**2):
                    if getattr(reader,'cancelled',None) is not None and reader.cancelled.is_set():break
                    if not chunk:continue
                    count+=len(chunk)
                    if count>expected:raise RuntimeError('Range response exceeds expected size')
                    self.wfile.write(chunk)
                if count!=expected:self.close_connection=True
            except (BrokenPipeError, ConnectionResetError): pass
            except Exception:
                # No provider URLs, credentials or exception strings in HTTP responses.
                if not sent_headers:self.send_error(502,'Drive range transport failed')
                self.close_connection = True
            finally:
                if response is not None: response.close()
                if session is not None: session.close()
    class RangeHTTPServer(ThreadingHTTPServer):request_queue_size=32
    server = RangeHTTPServer(('127.0.0.1',0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    return server, 'http://127.0.0.1:'+str(server.server_address[1])


def prepare_model_view(reader, run_root):
    model = Path(run_root) / 'main-model'; model.mkdir()
    for item in reader.items.values():
        path = model / item['name']
        if Path(item['name']).is_absolute() or '..' in Path(item['name']).parts:
            raise RuntimeError('Unsafe model manifest')
        path.parent.mkdir(parents=True,exist_ok=True)
        if item['name'].endswith('.safetensors'): path.write_bytes(b'ARC3STUB')
        else: path.write_bytes(reader.metadata(item['name']))
    index, shards = indexed_shards(model)
    if set(shards) != {item['name'] for item in reader.items.values() if item['name'].endswith('.safetensors')}:
        raise RuntimeError('Index does not match staged tensor manifest')
    return str(model)

def mount_auxiliary_with_retry(session, resource, persist):
    """Retry bounded virtual mounts; record progress before any failure escapes."""
    record = dict(resource, attempts=[], status='MOUNTING')
    persist(record)
    if resource_payload(resource).get('version') != 1:
        record['status'] = 'INVALID_RESOURCE_CONFIG'
        persist(record)
        return record
    deadline = time.monotonic() + 180
    terminal = {'AUTH_FAILED', 'ACCESS_DENIED', 'BACKEND_UNAVAILABLE',
                'BACKEND_ADDRESS_UNRECOGNIZED', 'INVALID_RESOURCE_CONFIG',
                'NOT_SUPPORTED', 'REDIRECT_BLOCKED', 'RESPONSE_TOO_LARGE',
                'UNEXPECTED_RESPONSE', 'MOUNT_RESPONSE_INVALID', 'MOUNT_PATH_INVALID'}
    def retryable(outcome):
        status = outcome['status']
        if status in terminal:
            return False
        if status == 'HTTP_ERROR':
            code = outcome.get('http_status')
            return code == 429 or (isinstance(code, int) and 500 <= code < 600)
        return status in {'TIMEOUT', 'CONNECTION_ERROR', 'REQUEST_ERROR',
                          'NOT_SUPPORTED_OR_NOT_FOUND', 'MOUNT_WAIT_TIMEOUT'}
    for number in range(1, 5):
        attempt = {'attempt': number, 'status': 'CHECKING_SUPPORT'}
        record['attempts'].append(attempt)
        persist(record)
        support, _ = endpoint_call(session, resource, 'is_supported')
        attempt['support'] = support
        record['support'] = support
        persist(record)
        # A support-endpoint 404 is ambiguous. Ask the pinned mount endpoint
        # directly; an explicit unsupported response or auth error is terminal.
        if support['status'] in {'SUPPORTED', 'NOT_SUPPORTED_OR_NOT_FOUND'}:
            attempt['direct_mount_after_support_404'] = support['status'] != 'SUPPORTED'
            attempt['status'] = 'REQUESTING_MOUNT'
            persist(record)
            before = disk_snapshot()
            outcome, root = endpoint_call(session, resource, 'mount')
            attempt['mount'] = outcome
            record['mount'] = outcome
            persist(record)
            if root is not None:
                outcome['path'] = str(root)
                persist(record)
                wait_until = min(deadline, time.monotonic() + MOUNT_TIMEOUT_SECONDS)
                while not root.is_dir() and time.monotonic() < wait_until:
                    time.sleep(1)
                if root.is_dir():
                    outcome['status'] = 'MOUNTED'
                    try:
                        if resource['kind'] == 'model':
                            outcome['metadata'] = inspect_model_metadata(root, resource)
                        else:
                            entries = []
                            for index, path in enumerate(root.iterdir()):
                                if index >= 100:
                                    break
                                entries.append({'name': path.name, 'is_directory': path.is_dir()})
                            outcome['top_level_entries'] = entries
                            outcome['file_contents_read'] = False
                    except OSError:
                        outcome['inspection_status'] = 'FILESYSTEM_ERROR'
                else:
                    outcome['status'] = 'MOUNT_WAIT_TIMEOUT'
            after = disk_snapshot()
            outcome.update(disk_before=before, disk_after=after,
                           local_used_delta_gib=round(after['used_gib']-before['used_gib'], 3))
        else:
            outcome = support
        attempt['status'] = outcome['status']
        record['status'] = outcome['status']
        persist(record)
        print(f"Mount: {resource['name']} | attempt {number}/4 | support {support['status']} (HTTP {support.get('http_status')}) | result {outcome['status']} (HTTP {outcome.get('http_status')})", flush=True)
        if outcome['status'] == 'MOUNTED' or not retryable(outcome):
            return record
        remaining = deadline-time.monotonic()
        if number == 4 or remaining <= 0:
            break
        time.sleep(min(2 ** number, remaining))
    record['retries_exhausted'] = True
    persist(record)
    return record


def mount_model_dependencies(session, resources, report):
    report['auxiliary_resources'] = []
    report['auxiliary_mounts'] = {}
    for resource in resources:
        slot = len(report['auxiliary_resources'])
        report['auxiliary_resources'].append(dict(resource, status='PENDING', attempts=[]))
        def persist(record):
            report['auxiliary_resources'][slot] = record
            save_report(report)
        try:
            record = mount_auxiliary_with_retry(session, resource, persist)
        except Exception as error:
            report['auxiliary_resources'][slot].update(status='MOUNT_HELPER_ERROR', error_type=type(error).__name__)
            report['failed_resource'] = resource['name']
            save_report(report)
            raise SetupFault(resource['name'].upper() + '_MOUNT_FAILED') from None
        if record.get('status') != 'MOUNTED':
            # Limit the fallback to the original small draft at version 1.
            # Access failures never trigger a download through another credential.
            can_download = (globals().get('ALLOW_DRAFT_DOWNLOAD_FALLBACK', False)
                            and resource['name'] == 'draft_model'
                            and record.get('status') in {'NOT_SUPPORTED','NOT_SUPPORTED_OR_NOT_FOUND',
                                                        'HTTP_ERROR','MOUNT_WAIT_TIMEOUT','TIMEOUT',
                                                        'CONNECTION_ERROR','BACKEND_UNAVAILABLE'})
            if can_download:
                try:
                    path = download_draft_v1(session, resource, record, persist)
                    record.update(status=record['download']['status'], resolved_path=path)
                    report['auxiliary_mounts'][resource['name']] = path
                    persist(record)
                    continue
                except Exception as error:
                    record.setdefault('download',{}).update(status='FAILED', error_type=type(error).__name__,
                                              error_code=getattr(error,'code','SEE_ERROR_TYPE'),
                                              http_status=getattr(error,'http_status',None))
                    report['failed_resource'] = resource['name']; persist(record)
                    raise SetupFault('DRAFT_MODEL_DOWNLOAD_FAILED') from None
            report['failed_resource'] = resource['name']
            save_report(report)
            raise SetupFault(resource['name'].upper() + '_MOUNT_FAILED')
        report['auxiliary_mounts'][resource['name']] = record['mount']['path']
        save_report(report)
    return report['auxiliary_mounts']

DRAFT_V1_PAYLOAD = {
    'ownerSlug': 'dfranzen', 'modelSlug': 'albucino-qwen3-8-flash-next-drafter',
    'framework': 'MODEL_FRAMEWORK_TRANSFORMERS', 'instanceSlug': 'default',
    'versionNumber': 1,
}
DRAFT_V1_HANDLE = 'dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'


def draft_rpc(session, method, payload):
    if method not in {'ListModelInstanceVersionFiles', 'DownloadModelInstanceVersion'}:
        raise ValueError('Unexpected draft operation')
    if any(payload.get(key) != value for key, value in DRAFT_V1_PAYLOAD.items()):
        raise TransferFault('DRAFT_RESOURCE_NOT_PINNED')
    return session.post('https://api.kaggle.com/v1/models.ModelApiService/' + method,
                        json=payload, stream=True, allow_redirects=False, timeout=(10, 60))


def draft_inventory(session):
    payload = dict(DRAFT_V1_PAYLOAD, pageSize=1000)
    listing, tokens = {}, set()
    for _ in range(20):
        with draft_rpc(session, 'ListModelInstanceVersionFiles', payload) as response:
            if response.status_code != 200:
                raise TransferFault('DRAFT_LIST_HTTP_' + str(response.status_code), response.status_code)
            raw = bytearray()
            for chunk in response.iter_content(8192):
                raw.extend(chunk)
                if len(raw) > 2*1024**2:
                    raise TransferFault('DRAFT_LIST_TOO_LARGE')
            data = json.loads(raw)
        entries = data.get('files')
        if not isinstance(entries, list) or data.get('errorMessage'):
            raise TransferFault('DRAFT_LIST_INVALID')
        for entry in entries:
            name, size = entry.get('name'), entry.get('size')
            if isinstance(size, str) and size.isdigit(): size = int(size)
            if not isinstance(name, str) or not name or not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise TransferFault('DRAFT_FILE_INVALID')
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or name in listing or str(path) != name or name == '.arc3-draft-download.json':
                raise TransferFault('DRAFT_PATH_INVALID')
            listing[name] = size
        if len(listing) > 2000 or sum(listing.values()) > MAX_DRAFT_DOWNLOAD_BYTES:
            raise TransferFault('DRAFT_SIZE_LIMIT')
        token = data.get('nextPageToken', '')
        if not token:
            if not listing: raise TransferFault('DRAFT_LIST_EMPTY')
            return listing
        if not isinstance(token, str) or token in tokens:
            raise TransferFault('DRAFT_PAGINATION_INVALID')
        tokens.add(token); payload['pageToken'] = token
    raise TransferFault('DRAFT_PAGE_LIMIT')


def validate_downloaded_draft(root):
    metadata = inspect_model_metadata(root, {'name':'draft_model'})
    if not any(row.get('status') == 'LAYOUT_OK' for row in metadata.get('checkpoints', [])):
        raise TransferFault('DRAFT_DOWNLOADED_LAYOUT_INVALID')
    return metadata


def download_draft_v1(session, resource, record, persist):
    if resource.get('name') != 'draft_model' or resource.get('handle') != DRAFT_V1_HANDLE:
        raise TransferFault('DRAFT_FALLBACK_RESOURCE_BLOCKED')
    download = {'status':'LISTING', 'handle':DRAFT_V1_HANDLE, 'version':1,
                'max_bytes':MAX_DRAFT_DOWNLOAD_BYTES, 'files':[],
                'integrity_note':'Pinned Kaggle file list and exact sizes; SHA256 computed locally, no independent Kaggle checksum supplied.'}
    record['download'] = download; persist(record)
    expected = draft_inventory(session)
    total = sum(expected.values())
    download.update(expected_bytes=total, file_count=len(expected), status='INVENTORY_OK'); persist(record)
    print(f'Draft direct download: version 1, {len(expected)} files, {total/1024**3:.2f} GiB.', flush=True)
    parent = Path(SETUP_ROOT).parent/'draft-cache-v1'; parent.mkdir(parents=True, exist_ok=True)
    cache = parent/'draft-download-v1'
    marker = cache/'.arc3-draft-download.json'
    if cache.exists():
        if cache.is_symlink() or not marker.is_file():
            raise TransferFault('DRAFT_EXISTING_CACHE_NOT_OWNED')
        stored = json.loads(marker.read_text())
        if stored.get('handle') != DRAFT_V1_HANDLE or stored.get('inventory') != expected:
            raise TransferFault('DRAFT_EXISTING_CACHE_MANIFEST_MISMATCH')
        for name, size in expected.items():
            path = cache/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                raise TransferFault('DRAFT_EXISTING_CACHE_INVALID')
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda:source.read(8*1024**2), b''): digest.update(chunk)
            if digest.hexdigest() != stored.get('sha256',{}).get(name):
                raise TransferFault('DRAFT_EXISTING_CACHE_CHECKSUM_MISMATCH')
        download.update(status='REUSED_VALIDATED_VERSION_1', path=str(cache), metadata=validate_downloaded_draft(cache))
        persist(record); return str(cache)
    reserve = max(MIN_LOCAL_FREE_GIB, MIN_INSTALL_START_FREE_GIB)
    if disk_snapshot()['free_gib'] < reserve + total/1024**3:
        raise TransferFault('DRAFT_INSUFFICIENT_DISK_HEADROOM')
    staging = Path(tempfile.mkdtemp(prefix='draft-v1-', dir=parent))
    hashes = {}; started = time.monotonic()
    try:
        with requests.Session() as public:
            public.trust_env = False
            for name, size in expected.items():
                download.update(status='DOWNLOADING', current_file=name); persist(record)
                payload = dict(DRAFT_V1_PAYLOAD, path=name)
                with draft_rpc(session, 'DownloadModelInstanceVersion', payload) as response:
                    if response.status_code not in {301,302,303,307,308}:
                        raise TransferFault('DRAFT_DOWNLOAD_HTTP_' + str(response.status_code), response.status_code)
                    url = urljoin('https://api.kaggle.com', response.headers.get('Location',''))
                path = staging/name; path.parent.mkdir(parents=True, exist_ok=True)
                count = 0; digest = hashlib.sha256(); progress = time.monotonic()
                with public_get(public, url, headers={'Accept-Encoding':'identity'}) as response:
                    if response.status_code != 200:
                        raise TransferFault('DRAFT_SIGNED_HTTP_' + str(response.status_code), response.status_code)
                    content_length = response.headers.get('Content-Length')
                    if content_length is not None and int(content_length) != size:
                        raise TransferFault('DRAFT_DOWNLOAD_SIZE_MISMATCH')
                    with path.open('xb') as output:
                        for chunk in response.iter_content(8*1024**2):
                            if not chunk: continue
                            count += len(chunk)
                            if count > size: raise TransferFault('DRAFT_DOWNLOAD_SIZE_MISMATCH')
                            if time.monotonic()-started > DRAFT_DOWNLOAD_TIMEOUT_SECONDS:
                                raise TransferFault('DRAFT_DOWNLOAD_TIMEOUT')
                            if disk_snapshot()['free_gib'] < MIN_LOCAL_FREE_GIB:
                                raise TransferFault('DRAFT_DOWNLOAD_DISK_GUARD')
                            digest.update(chunk); output.write(chunk)
                            if time.monotonic()-progress >= 30:
                                download['current_file_bytes_read'] = count; persist(record)
                                print(f'Draft: {name} | {count/1024**3:.2f}/{size/1024**3:.2f} GiB | free disk {disk_snapshot()["free_gib"]:.1f} GiB', flush=True)
                                progress = time.monotonic()
                if count != size: raise TransferFault('DRAFT_DOWNLOAD_SIZE_MISMATCH')
                hashes[name] = digest.hexdigest()
                download['files'].append({'name':name,'bytes':count,'local_sha256':hashes[name]}); persist(record)
        validate_downloaded_draft(staging)
        (staging/marker.name).write_text(json.dumps({'handle':DRAFT_V1_HANDLE,'inventory':expected,'sha256':hashes}))
        # Cache did not exist on entry. No pre-existing user directory is removed.
        if cache.exists(): raise TransferFault('DRAFT_CACHE_APPEARED_DURING_DOWNLOAD')
        staging.rename(cache)
        download.update(status='DOWNLOADED_VERSION_1', path=str(cache), metadata=validate_downloaded_draft(cache), elapsed_seconds=round(time.monotonic()-started,1))
        persist(record); return str(cache)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        download['partial_download_removed'] = True; persist(record)
        raise

import select


def recognized_trial_root(value):
    if not isinstance(value,str): return None
    root=Path(value)
    if not root.is_absolute() or root.parent != Path(SETUP_ROOT).parent:
        return None
    if not re.fullmatch(r'model-load-[0-9a-f]{12}',root.name): return None
    return root


def previous_failed_trial():
    path=Path(REPORT_PATH)
    if not path.is_file() or path.stat().st_size>2*1024**2: return None
    try:
        previous=json.loads(path.read_text())
        failed={'STOPPED_STARTUP_TIMEOUT','STOPPED_DISK_GUARD','STOPPED_RAM_GUARD',
                'SERVER_PROCESS_EXITED','PREPARATION_OR_LOAD_ERROR','INTERRUPTED',
                'INSUFFICIENT_START_RAM','PRIOR_RUN_PROCESSES_REMAIN'}
        if previous.get('test')!='ARC3 full-model staged loading trial' or previous.get('status') not in failed:
            return None
        value=previous.get('run_root') or previous.get('previous_failed_run')
        return value if recognized_trial_root(value) is not None else None
    except (ValueError,TypeError,OSError):
        return None


def cleanup_trial_processes(run_root, grace_seconds=10, force_seconds=5):
    """Signal only processes bearing this failed trial's exact inherited tags.

    Open pidfds before verifying /proc tags. Signals use pidfds, so an exited
    process's reused PID cannot cause an unrelated process to be terminated.
    No command lines, environment values or auth tokens enter the report.
    """
    root=recognized_trial_root(str(run_root))
    if root is None: return {'status':'UNRECOGNIZED_TRIAL_ROOT','matched_pids':[]}
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        return {'status':'PIDFD_NOT_AVAILABLE','matched_pids':[]}
    markers={
        b'ARC3_STAGED_LOADER=1',
        ('ARC3_STAGED_MODEL_DIR='+str(root/'main-model')).encode(),
        ('ARC3_STAGED_MANIFEST='+str(root/'manifest.json')).encode(),
    }
    targets=[]; errors=0
    try:
        for directory in Path('/proc').iterdir():
            if not directory.name.isdigit() or int(directory.name)==os.getpid(): continue
            pid=int(directory.name); fd=None
            try:
                fd=os.pidfd_open(pid,0)
                with (directory/'environ').open('rb') as source:
                    environment=source.read(512*1024)
                if not markers.issubset(set(environment.split(b'\0'))): continue
                targets.append((pid,fd));fd=None
            except (OSError,ProcessLookupError):
                pass
            finally:
                if fd is not None: os.close(fd)
        def alive():
            return [(pid,fd) for pid,fd in targets if not select.select([fd],[],[],0)[0]]
        def send(sig):
            nonlocal errors
            for pid,fd in alive():
                try: signal.pidfd_send_signal(fd,sig)
                except ProcessLookupError: pass
                except OSError: errors+=1
        def wait_for_exit(seconds):
            deadline=time.monotonic()+max(0,min(seconds,15))
            while alive() and time.monotonic()<deadline: time.sleep(0.2)
        send(signal.SIGTERM);wait_for_exit(grace_seconds)
        force=[pid for pid,fd in alive()]
        if force:
            send(signal.SIGKILL);wait_for_exit(force_seconds)
        remaining=[pid for pid,fd in alive()]
        return {'status':'CLEANED' if not remaining else 'PROCESSES_REMAIN',
                'matched_pids':[pid for pid,fd in targets], 'force_killed_pids':force,
                'remaining_pids':remaining,'signal_error_count':errors,
                'scope':'exact trial model-dir and manifest environment tags; pidfd signals only'}
    finally:
        for pid,fd in targets: os.close(fd)

def small_log_tail(path, limit=18000):
    path = Path(path)
    if not path.is_file(): return ''
    with path.open('rb') as source:
        source.seek(0,2); source.seek(max(0,source.tell()-limit))
        return source.read().decode(errors='replace')


def stage_summary(path):
    text = small_log_tail(path, 1024**2)
    records = []
    for line in text.splitlines():
        try: records.append(json.loads(line))
        except ValueError: pass
    timings={r['name']:r for r in records if r.get('event')=='stage_verified'}
    verified=set(timings)
    active = [r for r in records if r.get('event') in ('stage_start','stage_verified','stage_removed','stage_failed')]
    return {'verified_file_count':len(verified),'verified_names':sorted(verified),
            'verified_bytes':sum(r.get('bytes',0) for r in timings.values()),
            'shard_timings':list(timings.values()),
            'last_event':active[-1] if active else None,
            'hook_events':[r for r in records if r.get('event') in ('iterator_hook_installed','transport_hook_installed')]}


def cleanup_failed_staging(run_root):
    removed=0
    root=Path(run_root)/'main-model'
    if not root.is_dir():return {'removed_bytes':0}
    # Only files created by this trial under its unique model-view directory.
    for item in MODEL_MANIFEST:
        if not item['name'].endswith('.safetensors'):continue
        path=root/item['name'];partial=path.with_name(path.name+'.part')
        if partial.is_file() and not partial.is_symlink():
            removed+=partial.stat().st_size;partial.unlink()
        if path.is_file() and not path.is_symlink() and path.stat().st_size>8:
            removed+=path.stat().st_size;path.unlink();path.write_bytes(b'ARC3STUB')
    return {'removed_bytes':removed,'scope':'temporary files owned by this trial only'}


def observe_server(process, report, event_log, server_log):
    import urllib.request
    start = time.monotonic(); progress = start-30
    report['status'] = 'MODEL_LOADING'; save_report(report)
    peak = disk_snapshot()['used_gib']; minimum_ram = environment_snapshot()['ram_gib'].get('MemAvailable',0)
    try:
        while process.poll() is None:
            now = time.monotonic(); snapshot = environment_snapshot()
            peak = max(peak,snapshot['disk']['used_gib']); minimum_ram = min(minimum_ram,snapshot['ram_gib'].get('MemAvailable',0))
            if snapshot['disk']['free_gib'] < MIN_LOCAL_FREE_GIB:
                report['status']='STOPPED_DISK_GUARD'; break
            if snapshot['ram_gib'].get('MemAvailable',0) < MIN_AVAILABLE_RAM_GIB:
                report['status']='STOPPED_RAM_GUARD'; break
            if now-start > MODEL_STARTUP_TIMEOUT_SECONDS:
                report['status']='STOPPED_STARTUP_TIMEOUT'; break
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/health', timeout=2) as response:
                    if response.status==200:
                        report['status']='MODEL_HEALTHY'; break
            except Exception: pass
            if now-progress>=30:
                report['staging']=stage_summary(event_log)
                report['elapsed_seconds']=int(now-start)
                event=report['staging']['last_event']; label=(event['event']+' '+event['name']) if event else 'server starting'
                print(f"{int((now-start)/60)} min | {report['staging']['verified_file_count']}/38 files | free disk {snapshot['disk']['free_gib']:.1f} GiB | free RAM {snapshot['ram_gib'].get('MemAvailable',0):.1f} GiB | {label}",flush=True)
                save_report(report);progress=now
            time.sleep(2)
        if report['status']=='MODEL_LOADING': report['status']='SERVER_PROCESS_EXITED'
    except BaseException:
        report['status']='INTERRUPTED'; stop_owned_process(process); save_report(report); raise
    if report['status']!='MODEL_HEALTHY': stop_owned_process(process)
    report.update(staging=stage_summary(event_log), server_log_tail=small_log_tail(server_log),
                  observed_peak_used_gib=peak, observed_min_available_ram_gib=minimum_ram,
                  elapsed_seconds=round(time.monotonic()-start,1), environment_after=environment_snapshot())
    save_report(report)


def run_model_load():
    global MODEL_PROCESS, DRIVE_PROXY_SERVER
    trial_start=time.monotonic()
    previous_root=previous_failed_trial()
    report={'test':'ARC3 full-model staged loading trial','version':6,'status':'PREPARING',
            'utc':datetime.now(timezone.utc).isoformat(),'environment':environment_snapshot(),
            'inference_settings':CFG,'inference_settings_changed':False,
            'loader_transport_changed':True,'drive_fuse_used_for_main_weights':False,
            'weight_download_connections':WEIGHT_DOWNLOAD_CONNECTIONS,
            'shard_download_timeout_seconds':SHARD_DOWNLOAD_TIMEOUT_SECONDS,
            'source_model_handle':'dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1',
            'algorithm_or_prompt_test_started':False,'drive_quota_checked':False,
            'startup_timeout_seconds':MODEL_STARTUP_TIMEOUT_SECONDS,
            'previous_failed_run':previous_root}
    save_report(report)
    if previous_root is not None:
        print('Checking for leftover processes from a previous failed attempt.',flush=True)
        report['previous_trial_cleanup']=cleanup_trial_processes(previous_root)
        report['environment']=environment_snapshot();save_report(report)
        if report['previous_trial_cleanup'].get('remaining_pids'):
            report['status']='PRIOR_RUN_PROCESSES_REMAIN';save_report(report);return report
    # Avoid replacing a live server or killing unrelated processes.
    with socket.socket() as probe:
        if probe.connect_ex((SERVING_HOST,SERVED_MODEL_PORT))==0:
            report['status']='PORT_ALREADY_OCCUPIED';save_report(report);return report
    if report['environment']['ram_gib'].get('MemAvailable',0)<MIN_START_AVAILABLE_RAM_GIB:
        report['status']='INSUFFICIENT_START_RAM';save_report(report);return report
    run_root=Path(SETUP_ROOT).parent/('model-load-'+uuid.uuid4().hex[:12]);run_root.mkdir(parents=True)
    report['run_root']=str(run_root)
    server=None;process=None
    try:
        report['phase']='DRIVE_AUTHENTICATION'; save_report(report)
        print('Connecting to the Drive API. Choose the account that holds the model.',flush=True)
        from google.colab import auth
        import google.auth
        auth.authenticate_user()
        credentials,_=google.auth.default()
        reader=DriveReader(credentials,MODEL_MANIFEST)
        report['phase']='DRIVE_CONFIG_READ';save_report(report)
        config_data=reader.metadata('config.json',limit=1024**2)
        report['drive_access']={'status':'DRIVE_ACCESS_CONFIRMED','bytes':len(config_data),
                                'sha256':hashlib.sha256(config_data).hexdigest()}
        del config_data
        report['phase']='AUXILIARY_RESOURCES';save_report(report)
        print('Verifying the wheelhouse and draft model on Drive.',flush=True)
        if globals().get('ARC3_DRIVE_DATASETS_ONLY',False):
            mounts=mount_model_dependencies(None,[RESOURCES[1],RESOURCES[0]],report)
        else:
            session,mode=authenticated_session(); mounts={}
            try:
                # Resolve the package mount before spending disk/time on draft download.
                mounts=mount_model_dependencies(session,[RESOURCES[1],RESOURCES[0]],report)
            finally:
                session.headers.pop('Authorization',None);session.auth=None;session.close()
        report['auxiliary_mounts']=mounts;save_report(report)
        venv=Path(SETUP_ROOT)/'venv'; python=str(venv/'bin/python')
        if not Path(python).is_file():
            print('Fresh runtime: preparing the same pinned Python 3.12 environment.',flush=True)
            setup_report={'test':'model-load environment bootstrap','status':'PREPARING'}
            prepare_python312(setup_report,mounts['wheelhouse'])
            report['environment_bootstrap']=setup_report;save_report(report)
            if setup_report.get('runtime_check',{}).get('status')!='COMMAND_OK': raise SetupFault('ENVIRONMENT_BOOTSTRAP_FAILED')
        env=runtime_environment(SETUP_ROOT,python)
        report['phase']='MAIN_MODEL_METADATA';save_report(report)
        print('Preparing small model metadata files with checksums.',flush=True)
        model_dir=prepare_model_view(reader,run_root)
        report['model_dir']=model_dir
        report['phase']='TRANSPORT_PREPARATION';save_report(report)
        manifest_path=run_root/'manifest.json';manifest_path.write_text(json.dumps(MODEL_MANIFEST))
        plugins=run_root/'transport';plugins.mkdir()
        (plugins/'arc3_staged_loader.py').write_text(STAGED_LOADER_SOURCE)
        (plugins/'sitecustomize.py').write_text('import arc3_staged_loader\narc3_staged_loader.install_from_environment()\n')
        token=uuid.uuid4().hex+uuid.uuid4().hex
        server,proxy=start_drive_proxy(reader,token);DRIVE_PROXY_SERVER=server
        event_log=run_root/'staging.jsonl';event_log.touch()
        namespace=dict(globals(),MODEL_DIR=model_dir,DRAFT_MODEL_DIR=mounts['draft_model'],
                       WHEELHOUSE_DIR=mounts['wheelhouse'],PREFIX=SETUP_ROOT,
                       VENV=str(venv),SGLANG=str(venv/'bin/sglang'),env=env,
                       CUDA_HOME=env['CUDA_HOME'],CC=env['CC'],CXX=env['CXX'],cache=str(Path(SETUP_ROOT)/'cache'))
        report['phase']='DRAFT_METADATA';save_report(report)
        namespace['DRAFT_VIEW']=prepare_draft_view(model_dir,mounts['draft_model'],SETUP_ROOT)
        report['draft_view']=str(namespace['DRAFT_VIEW'])
        report['phase']='SERVER_ARGUMENTS';save_report(report)
        exec(ORIGINAL_RUNTIME_FLAGS,namespace)
        env=namespace['env']
        env.update({'ARC3_STAGED_LOADER':'1','ARC3_STAGED_MODEL_DIR':model_dir,
                    'ARC3_STAGED_MANIFEST':str(manifest_path),'ARC3_STAGED_PROXY':proxy,
                    'ARC3_STAGED_TOKEN':token,'ARC3_STAGED_LOG':str(event_log),
                    'ARC3_STAGED_FREE_FLOOR':str(MIN_LOCAL_FREE_GIB),
                    'ARC3_STAGED_CONNECTIONS':str(WEIGHT_DOWNLOAD_CONNECTIONS),
                    'ARC3_STAGED_SHARD_TIMEOUT':str(SHARD_DOWNLOAD_TIMEOUT_SECONDS),
                    'PYTHONPATH':str(plugins)})
        exec(ORIGINAL_ARGUMENTS,namespace)
        args=namespace['args'];report['serving_arguments']=args
        report['model_dir']=model_dir;report['server_log_path']=str(run_root/'serve.log')
        report['python_executable']=python
        save_report(report)
        print(f'Starting full model load; each shard is downloaded over {WEIGHT_DOWNLOAD_CONNECTIONS} connections and verified with a full SHA256.',flush=True)
        report['phase']='FULL_MODEL_LOADING';save_report(report)
        with (run_root/'serve.log').open('wb') as output:
            process=subprocess.Popen(args,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        MODEL_PROCESS=process; report['server_pid']=process.pid
        report['server_started_utc']=datetime.now(timezone.utc).isoformat()
        observe_server(process,report,event_log,run_root/'serve.log')
        if report['status']=='MODEL_HEALTHY':
            expected_names={item['name'] for item in MODEL_MANIFEST if item['name'].endswith('.safetensors')}
            if set(report['staging']['verified_names']) != expected_names:
                stop_owned_process(process)
                raise SetupFault('HEALTH_REPORTED_WITHOUT_ALL_VERIFIED_SHARDS')
            import urllib.request
            with urllib.request.urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/v1/models',timeout=10) as response:
                report['served_models']=json.loads(response.read(65536))
            report['server_kept_running']=True
            print('Model server ready. The ARC games/benchmark have not started yet.',flush=True)
        else:
            print('Load result:',report['status'],flush=True)
            print(report.get('server_log_tail','')[-7000:])
    except KeyboardInterrupt:
        if process is not None:stop_owned_process(process)
        report['status']='INTERRUPTED'
    except Exception as error:
        if process is not None:stop_owned_process(process)
        report['status']='PREPARATION_OR_LOAD_ERROR';report['error_type']=type(error).__name__
        report['error_code']=getattr(error,'code','SEE_RUNTIME_OR_SERVER_LOG')
        # Frame locations only: omit exception text, source lines and locals,
        # which can contain credentials or signed provider URLs.
        report['error_frames']=[{'file':Path(frame.filename).name,'line':frame.lineno,
                                 'function':frame.name}
                                for frame in traceback.extract_tb(error.__traceback__)[-12:]]
        if isinstance(error,NameError) and isinstance(error.name,str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',error.name):
            report['undefined_name']=error.name
            report['error_code']='UNDEFINED_NAME_'+error.name
        report['server_log_tail']=small_log_tail(run_root/'serve.log')
        print('Error:',report['error_code'],report['error_type']);print(report['server_log_tail'][-7000:])
    finally:
        if server is not None:
            if getattr(reader,'cancelled',None) is not None:reader.cancelled.set()
            server.shutdown();server.server_close()
        if report['status']!='MODEL_HEALTHY':
            report['owned_process_cleanup']=cleanup_trial_processes(str(run_root))
            report['temporary_staging_cleanup']=cleanup_failed_staging(run_root)
        report['environment_after']=environment_snapshot()
        report['total_elapsed_seconds']=round(time.monotonic()-trial_start,1)
        save_report(report)
    print('Report:',REPORT_PATH)
    return report



from urllib.parse import urljoin
from contextlib import contextmanager
from itertools import chain as _competition_chain
import re as _competition_re

COMPETITION_API_ROOT='https://api.kaggle.com/v1/competitions.CompetitionApiService/'


def competition_error_message(data,session):
    # Store only a bounded server message, never headers, credentials or signed URLs.
    message=data.get('message',data.get('errorMessage','')) if isinstance(data,dict) else ''
    if not message and isinstance(data,dict) and isinstance(data.get('error'),dict):
        message=data['error'].get('message','')
    if not isinstance(message,str):return ''
    message=_competition_re.sub(r'https?://[^\s<>"\']+','[URL]',message)
    message=_competition_re.sub(r'(?i)(?:Bearer|Basic)\s+\S+|KGAT_[A-Za-z0-9_-]+','[REDACTED]',message)
    auth=getattr(session,'headers',{}).get('Authorization','')
    secrets=[auth,auth.split(' ',1)[-1]] if auth else []
    if isinstance(getattr(session,'auth',None),(tuple,list)):secrets.extend(session.auth)
    for value in secrets:
        if isinstance(value,str) and value:message=message.replace(value,'[REDACTED]')
    return ' '.join(message.split())[:500]


def competition_small_json(response):
    if 'json' not in response.headers.get('Content-Type','').lower():return {}
    raw=bytearray()
    for chunk in response.iter_content(8192):
        raw.extend(chunk)
        if len(raw)>64*1024:return {}
    try:
        data=json.loads(raw)
        return data if isinstance(data,dict) else {}
    except (ValueError,UnicodeError):return {}


def competition_raise_http(stage,status,name,state,persist,message=''):
    code=stage+'_HTTP_'+str(status)
    state.update(status='FAILED',error_code=code,http_status=status,current_file=name,server_message=message)
    persist()
    print(f'Competition download error: {code} | file: {name or "file list"}',flush=True)
    if message:print('Kaggle message:',message,flush=True)
    if status==401:print('Kaggle API authentication rejected; check the kaggle-key secret.',flush=True)
    elif status==403:print('Kaggle access denied; check that the token account has joined the competition and accepted its rules.',flush=True)
    raise TransferFault(code,status)


def competition_rpc(session,method,name,state,persist):
    payload={'competitionName':'arc-prize-2026-arc-agi-3'}
    if method=='DownloadDataFile':payload['fileName']=name
    elif method=='ListDataFiles':payload.update(name)
    else:raise ValueError('Unexpected competition operation')
    display_name=name if isinstance(name,str) else ''
    for attempt in range(1,4):
        event={'method':method,'file':display_name,'attempt':attempt}
        state.setdefault('requests',[]).append(event)
        try:
            response=session.post(COMPETITION_API_ROOT+method,json=payload,
                timeout=(10,60),stream=True,allow_redirects=False)
        except (requests.Timeout,requests.ConnectionError) as error:
            event.update(status='NETWORK_ERROR',http_status=None,error_type=type(error).__name__);persist()
            if attempt==3:
                state.update(status='FAILED',error_code='COMPETITION_NETWORK_ERROR',http_status=None,current_file=display_name)
                persist();raise TransferFault('COMPETITION_NETWORK_ERROR') from None
            print(f'Retrying competition connection: {attempt}/3 | {display_name}',flush=True)
            time.sleep(2**attempt);continue
        status=response.status_code
        event.update(status='RESPONSE',http_status=status);persist()
        if status==429 or 500<=status<600:
            message=competition_error_message(competition_small_json(response),session)
            event['server_message']=message
            delay=response.headers.get('Retry-After','')
            delay=min(30,max(1,int(delay))) if delay.isdigit() else 2**attempt
            response.close()
            if attempt==3:competition_raise_http('COMPETITION_API',status,display_name,state,persist,message)
            print(f'Retrying competition request: HTTP {status} | {attempt}/3 | {display_name}',flush=True)
            time.sleep(delay);continue
        return response


@contextmanager
def competition_file_stream(session,public,name,state,persist):
    with competition_rpc(session,'DownloadDataFile',name,state,persist) as response:
        status=response.status_code
        content_type=response.headers.get('Content-Type','').lower()
        url=None
        if status in {301,302,303,307,308}:
            location=response.headers.get('Location','')
            if not location:raise TransferFault('COMPETITION_REDIRECT_LOCATION_MISSING',status)
            url=urljoin('https://api.kaggle.com',location)
        elif status==200 and 'json' in content_type:
            iterator=response.iter_content(8192);prefix=bytearray();complete=True
            for chunk in iterator:
                prefix.extend(chunk)
                if len(prefix)>64*1024:complete=False;break
            try:data=json.loads(prefix) if complete else {}
            except (ValueError,UnicodeError):data={}
            if not isinstance(data,dict):data={}
            api_code=data.get('code')
            if isinstance(api_code,int) and api_code>=400:
                competition_raise_http('COMPETITION_API',api_code,name,state,persist,competition_error_message(data,session))
            if isinstance(data.get('url'),str) and data['url']:
                # Some APIs serialize their HttpRedirect instead of emitting a 302.
                url=data['url']
            elif name.endswith(('.json','.jsonl')) and not data.get('errorMessage'):
                state.update(last_media_http_status=200,media_transport='DIRECT_API_STREAM');persist()
                yield _competition_chain([bytes(prefix)],iterator)
                return
            else:
                message=competition_error_message(data,session)
                state.update(server_message=message,http_status=status);persist()
                raise TransferFault('COMPETITION_UNEXPECTED_JSON_RESPONSE',status)
        elif status==200 and 'html' not in content_type:
            # A successful direct media response is valid too; no redirect required.
            state.update(last_media_http_status=200,media_transport='DIRECT_API_STREAM');persist()
            yield response.iter_content(1024**2)
            return
        else:
            message=competition_error_message(competition_small_json(response),session)
            competition_raise_http('COMPETITION_DOWNLOAD',status,name,state,persist,message)
        with public_get(public,url,headers={'Accept-Encoding':'identity'}) as media:
            state.update(last_media_http_status=media.status_code,media_transport='SIGNED_URL_STREAM');persist()
            if media.status_code!=200:
                competition_raise_http('COMPETITION_MEDIA',media.status_code,name,state,persist)
            if 'html' in media.headers.get('Content-Type','').lower():
                raise TransferFault('COMPETITION_UNEXPECTED_HTML_MEDIA',media.status_code)
            yield media.iter_content(1024**2)


def competition_layout(root):
    root=Path(root)
    return (root/'arc_agi_3_wheels').is_dir() and any((root/'arc_agi_3_wheels').glob('arc_agi-*.whl')) and (root/'environment_files').is_dir() and any((root/'environment_files').rglob('*.py'))


def competition_file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def competition_listing(session, cap, state=None, persist=None):
    state={} if state is None else state
    persist=(lambda:None) if persist is None else persist
    entries={};seen=set();payload={'competitionName':'arc-prize-2026-arc-agi-3','pageSize':1000}
    for page in range(100):
        with competition_rpc(session,'ListDataFiles',payload,state,persist) as response:
            if response.status_code!=200:
                competition_raise_http('COMPETITION_LIST',response.status_code,'',state,persist,
                    competition_error_message(competition_small_json(response),session))
            raw=bytearray()
            for chunk in response.iter_content(8192):
                raw.extend(chunk)
                if len(raw)>4*1024**2:raise TransferFault('COMPETITION_LIST_TOO_LARGE')
            data=json.loads(raw)
        if not isinstance(data.get('files'),list):raise TransferFault('COMPETITION_LIST_INVALID')
        for item in data['files']:
            name=item.get('name');size=item.get('totalBytes')
            if isinstance(size,str) and size.isdigit():size=int(size)
            if not isinstance(name,str) or not isinstance(size,int) or isinstance(size,bool) or size<0:
                raise TransferFault('COMPETITION_FILE_INVALID')
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:raise TransferFault('COMPETITION_PATH_INVALID')
            if not (name.startswith(('arc_agi_3_wheels/','environment_files/')) or name in {'arc_agi_3_wheels.zip','environment_files.zip'}):continue
            if name in entries:raise TransferFault('COMPETITION_DUPLICATE_FILE')
            entries[name]=size
            if len(entries)>20000 or sum(entries.values())>cap:raise TransferFault('COMPETITION_SIZE_LIMIT')
        token=data.get('nextPageToken','')
        if not token:
            if not entries:raise TransferFault('COMPETITION_REQUIRED_FILES_NOT_LISTED')
            return entries
        if not isinstance(token,str) or token in seen:raise TransferFault('COMPETITION_PAGE_TOKEN_INVALID')
        seen.add(token);payload['pageToken']=token
    raise TransferFault('COMPETITION_PAGE_LIMIT')


def restore_competition(root, session, cap, free_floor, report=None):
    state={'status':'PREPARING','competition':'arc-prize-2026-arc-agi-3','requests':[],'files_complete':0}
    if report is not None:report['competition_transfer']=state
    def persist():
        if report is not None:save_report(report)
    persist()
    root=Path(root);marker=root/'arc3-competition-cache.json'
    if marker.is_file() and competition_layout(root):
        info=json.loads(marker.read_text())
        if info.get('competition')=='arc-prize-2026-arc-agi-3':
            valid=True
            for record in info.get('files',[]):
                p=Path(record['name'])
                if p.is_absolute() or '..' in p.parts:valid=False;break
                path=root/p
                if not path.is_file() or path.is_symlink() or path.stat().st_size!=record['bytes'] or competition_file_sha256(path)!=record['sha256']:
                    valid=False;break
            if valid and info.get('files'):
                state.update(status='REUSED_VALIDATED',files_complete=len(info['files']));persist()
                return str(root)
    if root.exists():raise TransferFault('COMPETITION_CACHE_NOT_REUSABLE')
    expected=competition_listing(session,cap,state,persist)
    state.update(status='DOWNLOADING',file_count=len(expected),listed_bytes=sum(expected.values()));persist()
    if shutil.disk_usage('/content').free < free_floor*1024**3+2*cap:raise TransferFault('COMPETITION_DISK_HEADROOM')
    root.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='competition-download-',dir=root.parent));tree=staging/'tree';tree.mkdir()
    start=time.monotonic();network_bytes=0
    try:
        with requests.Session() as public:
            public.trust_env=False
            for number,(name,size) in enumerate(expected.items(),1):
                state.update(current_file=name);persist()
                print(f'Competition data: {number}/{len(expected)} | {name}',flush=True)
                download=staging/'download.part';count=0
                with competition_file_stream(session,public,name,state,persist) as chunks,download.open('xb') as output:
                    for chunk in chunks:
                        count+=len(chunk);network_bytes+=len(chunk)
                        if count>max(size*2,size+1024**2) or network_bytes>cap+64*1024**2:raise TransferFault('COMPETITION_DOWNLOAD_SIZE_LIMIT')
                        if time.monotonic()-start>1800:raise TransferFault('COMPETITION_DOWNLOAD_TIMEOUT')
                        if shutil.disk_usage('/content').free<free_floor*1024**3:raise TransferFault('COMPETITION_DISK_GUARD')
                        output.write(chunk)
                if name.endswith('.zip'):
                    with zipfile.ZipFile(download) as archive:
                        members=[m for m in archive.infolist() if not m.is_dir()]
                        if sum(m.file_size for m in members)>cap:raise TransferFault('COMPETITION_EXTRACT_SIZE_LIMIT')
                        for member in members:
                            p=Path(member.filename)
                            if p.is_absolute() or '..' in p.parts or '\\' in member.filename or stat.S_ISLNK(member.external_attr>>16):raise TransferFault('COMPETITION_UNSAFE_ARCHIVE')
                            if p.parts[0] not in {'arc_agi_3_wheels','environment_files'}:
                                p=Path(name[:-4])/p
                            target=tree/p;target.parent.mkdir(parents=True,exist_ok=True)
                            if target.exists():raise TransferFault('COMPETITION_DUPLICATE_ARCHIVE_FILE')
                            with archive.open(member) as source,target.open('xb') as output:shutil.copyfileobj(source,output,1024**2)
                else:
                    target=tree/name;target.parent.mkdir(parents=True,exist_ok=True)
                    if count==size:download.replace(target)
                    elif zipfile.is_zipfile(download):
                        with zipfile.ZipFile(download) as archive:
                            members=[m for m in archive.infolist() if not m.is_dir()]
                            if len(members)!=1 or members[0].filename not in {name,Path(name).name} or members[0].file_size!=size:
                                raise TransferFault('COMPETITION_SINGLE_FILE_ARCHIVE_MISMATCH')
                            with archive.open(members[0]) as source,target.open('xb') as output:shutil.copyfileobj(source,output,1024**2)
                    else:raise TransferFault('COMPETITION_FILE_SIZE_MISMATCH')
                download.unlink(missing_ok=True)
                if sum(p.stat().st_size for p in tree.rglob('*') if p.is_file())>cap:raise TransferFault('COMPETITION_EXTRACT_SIZE_LIMIT')
                state.update(files_complete=number,network_bytes=network_bytes);persist()
        if not competition_layout(tree):raise TransferFault('COMPETITION_LAYOUT_INVALID')
        records=[]
        for path in sorted(tree.rglob('*')):
            if path.is_file():
                digest=hashlib.sha256()
                with path.open('rb') as source:
                    for chunk in iter(lambda:source.read(8*1024**2),b''):digest.update(chunk)
                records.append({'name':str(path.relative_to(tree)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
        (tree/'arc3-competition-cache.json').write_text(json.dumps({'competition':'arc-prize-2026-arc-agi-3','files':records}))
        tree.replace(root)
        state.update(status='READY',path=str(root));persist()
        return str(root)
    except BaseException as error:
        state.update(status='FAILED',error_code=getattr(error,'code',str(error)),
            http_status=getattr(error,'http_status',state.get('http_status')),
            error_type=type(error).__name__)
        persist()
        raise
    finally:shutil.rmtree(staging)

# Pinned auxiliary packages only. Main model weights still use the Drive range loader.
WHEELHOUSE_V1_HANDLE = 'dfranzen/pennyroyal-v253'
WHEELHOUSE_V1_PAYLOAD = {'ownerSlug':'dfranzen', 'datasetSlug':'pennyroyal-v253', 'datasetVersionNumber':1}
WHEELHOUSE_MAX_BYTES = 64 * 1024**3
WHEELHOUSE_MAX_FILE_BYTES = 8 * 1024**3
WHEELHOUSE_MARKER = '.arc3-wheelhouse-v1.json'


def wheelhouse_safe_name(name):
    if (not isinstance(name,str) or not name or '\\' in name or '\x00' in name
            or Path(name).is_absolute() or '..' in Path(name).parts or str(Path(name)) != name
            or any(part.startswith('.') for part in Path(name).parts)):
        raise TransferFault('WHEELHOUSE_UNSAFE_PATH')
    return name


def wheelhouse_validate_inventory(listing):
    if not isinstance(listing,dict) or not listing or len(listing)>2000:
        raise TransferFault('WHEELHOUSE_INVENTORY_INVALID')
    for name,size in listing.items():
        wheelhouse_safe_name(name)
        if not isinstance(size,int) or isinstance(size,bool) or not 0<=size<=WHEELHOUSE_MAX_FILE_BYTES:
            raise TransferFault('WHEELHOUSE_FILE_SIZE_LIMIT')
    if sum(listing.values())>WHEELHOUSE_MAX_BYTES:
        raise TransferFault('WHEELHOUSE_TOTAL_SIZE_LIMIT')
    if 'requirements.lock' not in listing or not any(n.startswith('wheels/') and n.endswith('.whl') for n in listing):
        raise TransferFault('WHEELHOUSE_LAYOUT_INVALID')


def wheelhouse_inventory(session):
    payload=dict(WHEELHOUSE_V1_PAYLOAD,pageSize=1000)
    listing,tokens={},set()
    for _ in range(20):
        with dataset_rpc(session,'ListDatasetFiles',payload) as response:
            if response.status_code!=200:raise TransferFault('WHEELHOUSE_LIST_HTTP',response.status_code)
            raw=bytearray()
            for chunk in response.iter_content(8192):
                raw.extend(chunk)
                if len(raw)>2*1024**2:raise TransferFault('WHEELHOUSE_LIST_TOO_LARGE')
            data=json.loads(raw)
        entries=data.get('datasetFiles',data.get('files'))
        if not isinstance(entries,list) or data.get('errorMessage'):raise TransferFault('WHEELHOUSE_LIST_INVALID')
        for entry in entries:
            name=wheelhouse_safe_name(entry.get('name'))
            size=entry.get('totalBytes')
            if isinstance(size,str) and size.isdigit():size=int(size)
            if name in listing:raise TransferFault('WHEELHOUSE_DUPLICATE_FILE')
            if not isinstance(size,int) or isinstance(size,bool) or not 0<=size<=WHEELHOUSE_MAX_FILE_BYTES:
                raise TransferFault('WHEELHOUSE_FILE_SIZE_LIMIT')
            listing[name]=size
        if len(listing)>2000 or sum(listing.values())>WHEELHOUSE_MAX_BYTES:
            raise TransferFault('WHEELHOUSE_TOTAL_SIZE_LIMIT')
        token=data.get('nextPageToken','')
        if not token:
            wheelhouse_validate_inventory(listing)
            return listing
        if not isinstance(token,str) or token in tokens:raise TransferFault('WHEELHOUSE_PAGINATION_INVALID')
        tokens.add(token);payload['pageToken']=token
    raise TransferFault('WHEELHOUSE_PAGE_LIMIT')


def wheelhouse_cache_path():
    configured=globals().get('_ARC3_WHEELHOUSE_CACHE_ROOT')
    if not configured:raise TransferFault('WHEELHOUSE_CACHE_PATH_MISSING')
    return Path(configured)


def wheelhouse_write_marker(cache,state):
    # Never overwrite a user's unrelated directory or file.
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='.arc3-state-',dir=cache,delete=False) as output:
        temporary=Path(output.name)
        json.dump(state,output,sort_keys=True)
    try:temporary.replace(cache/WHEELHOUSE_MARKER)
    finally:temporary.unlink(missing_ok=True)


def wheelhouse_copy_file(source,destination,size):
    count=0;digest=hashlib.sha256()
    with destination.open('xb') as output:
        for chunk in iter(lambda:source.read(8*1024**2),b''):
            count+=len(chunk)
            if count>size:raise TransferFault('WHEELHOUSE_FILE_SIZE_MISMATCH')
            if disk_snapshot()['free_gib']<MIN_LOCAL_FREE_GIB:raise TransferFault('WHEELHOUSE_DISK_GUARD')
            output.write(chunk);digest.update(chunk)
    if count!=size:raise TransferFault('WHEELHOUSE_FILE_SIZE_MISMATCH')
    return digest.hexdigest()


def wheelhouse_unpack_file(transfer,destination,name,size):
    # .whl files are ZIPs themselves. Unwrap only a single exact requested file.
    if zipfile.is_zipfile(transfer):
        with zipfile.ZipFile(transfer) as archive:
            members=[m for m in archive.infolist() if not m.is_dir()]
            if len(members)==1 and members[0].filename in {name,Path(name).name}:
                member=members[0]
                if member.file_size!=size or stat.S_ISLNK(member.external_attr>>16):
                    raise TransferFault('WHEELHOUSE_WRAPPER_INVALID')
                with archive.open(member) as source:
                    return wheelhouse_copy_file(source,destination,size)
    with transfer.open('rb') as source:
        return wheelhouse_copy_file(source,destination,size)


def download_wheelhouse_v1(session,resource,record,persist):
    if resource.get('name')!='wheelhouse' or resource.get('handle')!=WHEELHOUSE_V1_HANDLE or resource.get('version')!=1:
        raise TransferFault('WHEELHOUSE_RESOURCE_NOT_PINNED')
    cache=wheelhouse_cache_path();marker=cache/WHEELHOUSE_MARKER
    download={'status':'PREPARING','handle':WHEELHOUSE_V1_HANDLE,'version':1,'path':str(cache/'content'),
        'integrity_note':'Pinned Kaggle listing and exact sizes; local SHA256 for cache reuse, not an independent publisher checksum.'}
    record['download']=download;persist(record)
    if cache.exists():
        if cache.is_symlink() or not marker.is_file() or marker.is_symlink() or marker.stat().st_size>2*1024**2:
            raise TransferFault('WHEELHOUSE_CACHE_NOT_OWNED')
        state=json.loads(marker.read_text())
        if state.get('handle')!=WHEELHOUSE_V1_HANDLE or state.get('version')!=1:
            raise TransferFault('WHEELHOUSE_CACHE_IDENTITY_MISMATCH')
        expected=state.get('inventory');wheelhouse_validate_inventory(expected)
        if not isinstance(state.get('sha256'),dict) or set(state['sha256'])-set(expected):
            raise TransferFault('WHEELHOUSE_CACHE_STATE_INVALID')
    else:
        expected=wheelhouse_inventory(session)
        state={'handle':WHEELHOUSE_V1_HANDLE,'version':1,'inventory':expected,'sha256':{}}
        cache.mkdir(parents=True,exist_ok=False)
        wheelhouse_write_marker(cache,state)
    tree=cache/'content'
    if tree.is_symlink():raise TransferFault('WHEELHOUSE_UNSAFE_CACHE_PATH')
    tree.mkdir(exist_ok=True)
    download.update(status='VALIDATING_CACHE',expected_bytes=sum(expected.values()),file_count=len(expected),files_complete=0)
    persist(record)
    print(f'Wheelhouse v1: {len(expected)} files, {sum(expected.values())/1024**3:.2f} GiB. Drive cache: {tree}',flush=True)
    started=time.monotonic();downloaded=0
    with requests.Session() as public:
        public.trust_env=False
        for name,size in sorted(expected.items()):
            path=tree/name
            if any(p.is_symlink() for p in (path,*path.parents)):
                raise TransferFault('WHEELHOUSE_UNSAFE_CACHE_PATH')
            digest=state['sha256'].get(name)
            if path.exists():
                if not digest or not path.is_file() or path.stat().st_size!=size:
                    raise TransferFault('WHEELHOUSE_CACHE_FILE_INVALID')
                if sha256(path)!=digest:raise TransferFault('WHEELHOUSE_CACHE_CHECKSUM_MISMATCH')
            else:
                if disk_snapshot()['free_gib']<MIN_LOCAL_FREE_GIB+(size+64*1024**2)/1024**3:
                    raise TransferFault('WHEELHOUSE_INSUFFICIENT_DISK_HEADROOM')
                download.update(status='DOWNLOADING',current_file=name);persist(record)
                print('Downloading wheelhouse:',name,flush=True)
                payload=dict(WHEELHOUSE_V1_PAYLOAD,fileName=name,raw=True)
                with dataset_rpc(session,'DownloadDataset',payload) as response:
                    if response.status_code not in {301,302,303,307,308}:
                        raise TransferFault('WHEELHOUSE_DOWNLOAD_HTTP',response.status_code)
                    url=urljoin('https://api.kaggle.com',response.headers.get('Location',''))
                scratch=Path('/content/arc3-preflight');scratch.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='wheel-file-',dir=scratch) as temporary:
                    transfer=Path(temporary)/'transfer';count=0;progress=time.monotonic()
                    with public_get(public,url,headers={'Accept-Encoding':'identity'}) as response:
                        if response.status_code!=200:raise TransferFault('WHEELHOUSE_SIGNED_HTTP',response.status_code)
                        with transfer.open('xb') as output:
                            for chunk in response.iter_content(8*1024**2):
                                if not chunk:continue
                                count+=len(chunk)
                                if count>size+64*1024**2:raise TransferFault('WHEELHOUSE_TRANSFER_SIZE_LIMIT')
                                if time.monotonic()-started>2*60*60:raise TransferFault('WHEELHOUSE_DOWNLOAD_TIMEOUT')
                                if disk_snapshot()['free_gib']<MIN_LOCAL_FREE_GIB:raise TransferFault('WHEELHOUSE_DISK_GUARD')
                                output.write(chunk)
                                if time.monotonic()-progress>=30:
                                    print(f'Wheelhouse: {name} | {count/1024**2:.0f} MiB',flush=True)
                                    progress=time.monotonic()
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with tempfile.NamedTemporaryFile(prefix='.arc3-file-',dir=path.parent,delete=False) as output:
                        staged=Path(output.name)
                    staged.unlink()
                    try:
                        computed=wheelhouse_unpack_file(transfer,staged,name,size)
                        if digest and computed!=digest:raise TransferFault('WHEELHOUSE_CACHE_CHECKSUM_MISMATCH')
                        # Store digest first: an interruption before rename can safely retry this file.
                        state['sha256'][name]=computed;wheelhouse_write_marker(cache,state)
                        if path.exists():raise TransferFault('WHEELHOUSE_CACHE_FILE_APPEARED')
                        staged.rename(path)
                    finally:staged.unlink(missing_ok=True)
                downloaded+=1
            download['files_complete']+=1;persist(record)
    if not (tree/'wheels').is_dir() or not (tree/'requirements.lock').is_file():
        raise TransferFault('WHEELHOUSE_LAYOUT_INVALID')
    download.update(status='DOWNLOADED_VERSION_1' if downloaded else 'REUSED_VALIDATED_VERSION_1',
        files_downloaded=downloaded,elapsed_seconds=round(time.monotonic()-started,1))
    persist(record)
    return str(tree)

_ARC3_ORIGINAL_MOUNT_DEPENDENCIES = mount_model_dependencies


def wheelhouse_mount_allows_download(record):
    # Remote mount availability can differ from the pinned public download API.
    # Access failures and unexpected mount responses retain their original error.
    outcomes=[record,record.get('support',{}),record.get('mount',{})]
    for attempt in record.get('attempts',[]):
        outcomes.extend([attempt.get('support',{}),attempt.get('mount',{})])
    if any(x.get('status') in {'AUTH_FAILED','ACCESS_DENIED','INVALID_RESOURCE_CONFIG'}
           or x.get('http_status') in {401,403} for x in outcomes):return False
    status=record.get('status')
    if status=='HTTP_ERROR':
        code=record.get('mount',{}).get('http_status',record.get('support',{}).get('http_status'))
        return code==429 or isinstance(code,int) and 500<=code<600
    return status in {'NOT_SUPPORTED','NOT_SUPPORTED_OR_NOT_FOUND','MOUNT_WAIT_TIMEOUT',
                      'TIMEOUT','CONNECTION_ERROR','BACKEND_UNAVAILABLE'}


def mount_model_dependencies(session, resources, report):
    # Reuse the verified inputs prepared before CPU tests; no second mount request at model startup.
    cached = globals().get('_ARC3_RESOLVED_MOUNTS', {})
    resolved = dict(report.get('auxiliary_mounts',{}))
    names={r['name'] for r in resources}
    records = [r for r in report.get('auxiliary_resources',[]) if r.get('name') not in names]
    def direct_wheelhouse(resource,record):
        def persist(updated):
            report['auxiliary_resources']=records+[updated]
            report['auxiliary_mounts']=dict(resolved)
            save_report(report)
        try:
            path=download_wheelhouse_v1(session,resource,record,persist)
        except Exception as error:
            record.setdefault('download',{}).update(status='FAILED',error_type=type(error).__name__,
                error_code=getattr(error,'code','SEE_ERROR_TYPE'),http_status=getattr(error,'http_status',None))
            report['failed_resource']='wheelhouse';persist(record)
            print('Wheelhouse download error:',getattr(error,'code',type(error).__name__),
                  '| HTTP:',getattr(error,'http_status',None),flush=True)
            raise SetupFault('WHEELHOUSE_DOWNLOAD_FAILED') from None
        record.update(status=record['download']['status'],resolved_path=path)
        resolved['wheelhouse']=path;records.append(record)
        if report.get('failed_resource')=='wheelhouse':report.pop('failed_resource',None)
    for resource in resources:
        path = cached.get(resource['name'])
        valid = bool(path and Path(path).is_dir())
        if valid and resource['name'] == 'wheelhouse':
            valid = (Path(path)/'wheels').is_dir() and (Path(path)/'requirements.lock').is_file()
        if valid and resource['name'] == 'draft_model':
            metadata = inspect_model_metadata(Path(path),resource)
            valid = any(c.get('status')=='LAYOUT_OK' for c in metadata.get('checkpoints',[]))
        if valid:
            resolved[resource['name']] = path
            records.append(dict(resource,status='LOCAL_INPUT_REUSED',resolved_path=path,attempts=[]))
            print('Reusing ready input:',resource['name'],flush=True)
        else:
            cache_root=globals().get('_ARC3_WHEELHOUSE_CACHE_ROOT')
            if resource['name']=='wheelhouse' and cache_root and Path(cache_root).exists():
                direct_wheelhouse(resource,dict(resource,status='CHECKING_DRIVE_CACHE',attempts=[]))
            else:
                try:
                    fetched = _ARC3_ORIGINAL_MOUNT_DEPENDENCIES(session,[resource],report)
                except SetupFault:
                    record=next((r for r in report.get('auxiliary_resources',[]) if r.get('name')==resource['name']),{})
                    if resource['name']!='wheelhouse' or not wheelhouse_mount_allows_download(record):raise
                    print('Wheelhouse mount service unavailable; downloading the pinned v1 files directly.',flush=True)
                    direct_wheelhouse(resource,record)
                else:
                    resolved.update(fetched)
                    records.extend(report.get('auxiliary_resources',[]))
        report['auxiliary_resources'] = list(records)
        report['auxiliary_mounts'] = dict(resolved)
        save_report(report)
    return resolved
