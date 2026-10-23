"""Colab adapter for arc-agi-3-final-solution.ipynb.

Executed in the Colab kernel's namespace by the bootstrap cell (Colab only). It
1. reads the Drive configuration JSON (COLAB_CONFIG_PATH) and validates the inputs,
2. loads the model-loader helpers (colab_model_loader.py),
3. prepares the pinned Python 3.12 + SGLang environment and starts a Python 3.12
   harness process (the "bridge"),
4. forwards every later notebook cell to that process, so the same cells run on
   Kaggle (in the kernel) and on Colab (in the bridge). A cell whose first line is
   `# arc3-step: parent` stays in the Colab kernel; other `# arc3-step: <tag>`
   lines mark the cells that need Colab-side work (model, provenance, run, export).

Needs from the notebook: COLAB_CONFIG_PATH, HARNESS_DIR, HARNESS_INFO, NOTEBOOK_SUPPORT_DIR,
ARC3_BUNDLE_NAME and the names in ARC3_SHARED_SETTINGS.
"""
import hashlib as _config_hashlib
import json as _config_json
import re as _config_re
from pathlib import Path as _ConfigPath

ARC3_ADAPTER_INFO = {
    'adapter_version': 5,
    'harness_source': 'directory (HARNESS_DIR_COLAB)',
    'main_weights_origin': 'Drive API',
    'weight_download_connections': 8,
    'auxiliary_resource_versions': {'wheelhouse': 1, 'draft_model': 1, 'source_bundle': 1},
    'cell_forwarding': 'IPython input transformer -> Python 3.12 bridge',
}


def _arc3_validate_config(data):
    if not isinstance(data,dict) or set(data)!={'schema_version','inputs','outputs'} or data['schema_version']!=1:
        raise ValueError('Colab config requires schema_version=1, inputs and outputs.')
    inputs=data['inputs'];outputs=data['outputs']
    required={'main_model','wheelhouse','draft_model','source_bundle','competition'}
    if not isinstance(inputs,dict) or not required<=set(inputs) or set(inputs)-required-{'harness_patch','harness'}:
        raise ValueError('Colab config inputs are incomplete.')
    # The harness comes from HARNESS_DIR_COLAB in the notebook; old harness_patch/harness
    # entries are accepted and ignored so an existing Drive JSON keeps working.
    if not isinstance(outputs,dict) or set(outputs)!={'local_work_dir','drive_results_dir'}:
        raise ValueError('Colab config outputs require local_work_dir and drive_results_dir.')
    for name,value in outputs.items():
        if not isinstance(value,str) or not _ConfigPath(value).is_absolute() or '..' in _ConfigPath(value).parts:
            raise ValueError('Invalid output path: '+name)
    if _ConfigPath('/content/drive') in _ConfigPath(outputs['local_work_dir']).parents:
        raise ValueError('local_work_dir must be on the runtime local disk.')
    if _ConfigPath('/content/drive') not in _ConfigPath(outputs['drive_results_dir']).parents:
        raise ValueError('drive_results_dir must be inside /content/drive.')
    model=inputs['main_model']
    if not isinstance(model,dict) or set(model)!={'handle','files'} or model['handle']!='dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1':
        raise ValueError('Config must identify the original main checkpoint at version 1.')
    files=model['files'];names=set();content=[]
    if not isinstance(files,list) or len(files)!=50:raise ValueError('Main model manifest must contain all 50 original files.')
    for item in files:
        if not isinstance(item,dict) or set(item)!={'name','bytes','sha256','drive_file_id'}:
            raise ValueError('Invalid model file record.')
        name=item['name'];size=item['bytes'];sha=item['sha256'];file_id=item['drive_file_id']
        if not isinstance(name,str) or not name or _ConfigPath(name).is_absolute() or '..' in _ConfigPath(name).parts or '\\' in name or name in names:
            raise ValueError('Invalid/duplicate model filename.')
        if not isinstance(size,int) or isinstance(size,bool) or size<0 or not isinstance(sha,str) or not _config_re.fullmatch('[0-9a-f]{64}',sha):
            raise ValueError('Invalid model file size/SHA256.')
        if not isinstance(file_id,str) or not _config_re.fullmatch('[A-Za-z0-9_-]{10,200}',file_id):
            raise ValueError('Invalid Drive file ID.')
        names.add(name);content.append({key:item[key] for key in ('name','bytes','sha256')})
    fingerprint=_config_hashlib.sha256(_config_json.dumps(sorted(content,key=lambda x:x['name']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if fingerprint!='8b43c72e2bd9aa5bb64c6b57ebbc40653a8faa79bcf9c1030f7ff99c6b893cfa':raise ValueError('JSON file sizes/SHA256 values differ from the original checkpoint.')
    expected={
        'wheelhouse':'dfranzen/pennyroyal-v253',
        'draft_model':'dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1',
        'source_bundle':'dfranzen/taaf-kaggle-source-bundle-copy'}
    for name,handle in expected.items():
        item=inputs[name]
        if not isinstance(item,dict) or set(item)!={'handle','version','local_path'} or item['handle']!=handle or type(item['version']) is not int or item['version']!=1:
            raise ValueError('Config must preserve the original '+name+' at version 1.')
        path=item['local_path']
        if (not isinstance(path,str) or not _ConfigPath(path).is_absolute()
                or '..' in _ConfigPath(path).parts
                or (name!='draft_model' and _ConfigPath('/content/drive') not in _ConfigPath(path).parents)):
            raise ValueError('Colab dataset local_path must point to its Drive copy: '+name)
    item=inputs['competition']
    if (not isinstance(item,dict) or set(item)!={'name','local_path'}
            or item['name']!='arc-prize-2026-arc-agi-3'
            or not isinstance(item['local_path'],str)
            or not _ConfigPath(item['local_path']).is_absolute()
            or '..' in _ConfigPath(item['local_path']).parts
            or _ConfigPath('/content/drive') not in _ConfigPath(item['local_path']).parents):
        raise ValueError('Competition local_path must point to its Drive copy.')
    return data


def _arc3_load_colab_config():
    global _ARC3_COLAB_CONFIG
    from google.colab import auth,drive
    print('Google Drive authorization: choose the account that holds the model and the JSON config.',flush=True)
    auth.authenticate_user()
    drive.mount('/content/drive')
    path=_ConfigPath(COLAB_CONFIG_PATH)
    if not path.is_file():
        raise FileNotFoundError('Colab configuration file not found: '+str(path)+'. Save colab_config.json at this Drive path.')
    if path.stat().st_size>256*1024:raise ValueError('Colab configuration JSON is unexpectedly large.')
    _ARC3_COLAB_CONFIG=_arc3_validate_config(_config_json.loads(path.read_text()))
    inputs=_ARC3_COLAB_CONFIG['inputs'];outputs=_ARC3_COLAB_CONFIG['outputs']
    globals().update(
        COLAB_MAIN_MODEL_HANDLE=inputs['main_model']['handle'],COLAB_LLM_FILES=inputs['main_model']['files'],
        COLAB_WHEELHOUSE_DIR=inputs['wheelhouse']['local_path'],COLAB_DRAFT_MODEL_DIR=inputs['draft_model']['local_path'],
        COLAB_SOURCE_BUNDLE_DIR=inputs['source_bundle']['local_path'],COLAB_COMPETITION_DIR=inputs['competition']['local_path'],
        COLAB_WORKING_ROOT=outputs['local_work_dir'],COLAB_DRIVE_RESULTS_DIR=outputs['drive_results_dir'])
    print('Colab file configuration read from the Drive JSON file:',str(path),flush=True)


def _arc3_check_drive_api():
    import google.auth
    credentials,_=google.auth.default()
    reader=_ARC3_HELPERS['DriveReader'](credentials,_ARC3_HELPERS['MODEL_MANIFEST'])
    content=reader.metadata('config.json',limit=1024**2)
    data=_config_json.loads(content)
    if not isinstance(data,dict):raise ValueError('Main model config.json is invalid.')
    result={'status':'DRIVE_ACCESS_CONFIRMED','bytes':len(content),'sha256':_config_hashlib.sha256(content).hexdigest(),
        'model_weights_read':False,'before_kaggle_mounts':True}
    print('Drive API: model config.json size/SHA256/JSON verification passed. Starting Kaggle mounts.',flush=True)
    return result


_arc3_load_colab_config()

# ---------------------------------------------------------------- model loader helpers
print('Preparing Colab helpers; model loading has not started yet.',flush=True)
_ARC3_HELPERS = {"__name__": "arc3_colab_model_loader"}
_arc3_loader_path = _ConfigPath(NOTEBOOK_SUPPORT_DIR) / 'colab_model_loader.py'
exec(compile(_arc3_loader_path.read_text(encoding='utf-8'), str(_arc3_loader_path), 'exec'), _ARC3_HELPERS)
_ARC3_HELPERS['MODEL_MANIFEST'] = COLAB_LLM_FILES
COLAB_LLM_SETTINGS = dict(_ARC3_HELPERS['CFG'])
COLAB_RESOURCES = [dict(resource, handle=_ARC3_COLAB_CONFIG['inputs'][resource['name']]['handle'])
                   for resource in _ARC3_HELPERS['RESOURCES']]
_ARC3_HELPERS['RESOURCES'] = COLAB_RESOURCES
COLAB_MODEL_SETUP_ROOT = _ARC3_HELPERS['SETUP_ROOT']
COLAB_KAGGLE_SECRET_NAME = _ARC3_HELPERS['KAGGLE_SECRET_NAME']
COLAB_KAGGLE_AUTH_MODE = _ARC3_HELPERS['KAGGLE_AUTH_MODE']
COLAB_KAGGLE_USERNAME = _ARC3_HELPERS['KAGGLE_USERNAME']
COLAB_KAGGLE_USERNAME_SECRET_NAME = _ARC3_HELPERS['KAGGLE_USERNAME_SECRET_NAME']
COLAB_MODEL_REPORT_PATH = _ARC3_HELPERS['REPORT_PATH']
COLAB_RUNTIME_REPORT_PATH = '/content/arc3-preflight/colab_setup_report.json'
COLAB_WEIGHT_DOWNLOAD_CONNECTIONS = 8
COLAB_MODEL_STARTUP_TIMEOUT_SECONDS = 60 * 60
COLAB_ALLOW_HARNESS_PYPI = True
COLAB_REUSE_MODEL_SERVER = True
COLAB_LOAD_MODEL_IF_NEEDED = True

# ---------------------------------------------------------------- Python 3.12 bridge
import json as _bridge_json
import subprocess as _bridge_subprocess
import threading as _bridge_threading
import queue as _bridge_queue
import time as _bridge_time
import signal as _bridge_signal
from pathlib import Path as _BridgePath


class ARC3PythonBridge:
    def __init__(self, python, child_source, root, env):
        root = _BridgePath(root);root.mkdir(parents=True, exist_ok=True)
        path = root / 'python312_notebook_bridge.py';path.write_text(child_source)
        self.process = _bridge_subprocess.Popen([python, '-u', str(path)], env=env,
            stdin=_bridge_subprocess.PIPE, stdout=_bridge_subprocess.PIPE,
            stderr=_bridge_subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        self.messages = _bridge_queue.Queue()
        self.reader = _bridge_threading.Thread(target=self._read, daemon=True);self.reader.start()
        self.log = (root / 'notebook_console.log').open('a', buffering=1)
        self.failed = False
        deadline = _bridge_time.monotonic() + 60
        while True:
            event = self._next(deadline)
            if event.get('arc3_bridge_event') == 'ready':
                if not event['python'].startswith('3.12.'):
                    self.close();raise RuntimeError('ARC3 harness requires Python 3.12.')
                print('Colab harness Python:', event['python'], flush=True);break
            self._show(event)

    def _read(self):
        for line in self.process.stdout:
            try:
                data = _bridge_json.loads(line)
                if not isinstance(data, dict) or 'arc3_bridge_event' not in data:raise ValueError()
            except (ValueError, TypeError):
                data = {'arc3_bridge_event':'stream', 'name':'stdout', 'text':line}
            self.messages.put(data)
        self.messages.put({'arc3_bridge_event':'eof'})

    def _next(self, deadline=None):
        while True:
            if deadline is not None and _bridge_time.monotonic()>deadline:
                raise RuntimeError('Python 3.12 harness did not become ready.')
            try:return self.messages.get(timeout=1)
            except _bridge_queue.Empty:
                if self.process.poll() is not None:raise RuntimeError('Python 3.12 harness exited.')

    def _show(self, event):
        kind = event.get('arc3_bridge_event')
        if kind == 'stream':
            text = event['text'];print(text, end='', flush=True);self.log.write(text)
        elif kind == 'display':
            from IPython.display import display
            display(event['data'], raw=True, metadata=event.get('metadata',{}))
        elif kind == 'eof':raise RuntimeError('Python 3.12 harness exited.')

    def execute(self, code):
        if self.failed:raise RuntimeError('Previous harness cell failed. Start a new experiment run.')
        self.process.stdin.write(_bridge_json.dumps({'code':code}, ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        try:
            while True:
                event = self._next()
                if event.get('arc3_bridge_event') == 'done':
                    if not event['success']:
                        self.failed = True
                        raise RuntimeError('Python 3.12 cell failed: '+str(event.get('error_type')))
                    return
                self._show(event)
        except KeyboardInterrupt:
            self.failed = True
            if self.process.poll() is None:self.process.send_signal(_bridge_signal.SIGINT)
            raise

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"action":"close"}\n');self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, _bridge_subprocess.TimeoutExpired):
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except _bridge_subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        self.process.stdin.close();self.process.stdout.close();self.log.close()

# ---------------------------------------------------------------- runtime orchestration
import json as _arc3_json
import re as _arc3_re
import socket as _arc3_socket
import subprocess as _arc3_subprocess
import uuid as _arc3_uuid
import shutil as _arc3_shutil
from pathlib import Path as _ARC3Path
from urllib.request import urlopen as _arc3_urlopen


def _arc3_previous_report():
    path = _ARC3Path(COLAB_MODEL_REPORT_PATH)
    if path.is_file():
        try:return _arc3_json.loads(path.read_text())
        except (ValueError, OSError):pass
    return {}


def _arc3_valid_server(report):
    """Reuse a running model server only if it is exactly the one this notebook would start."""
    if report.get('status') != 'MODEL_HEALTHY':return False
    if report.get('inference_settings') != _ARC3_HELPERS['CFG']:
        raise RuntimeError('Existing server model settings differ from this notebook.')
    if report.get('source_model_handle') != COLAB_MAIN_MODEL_HANDLE:
        raise RuntimeError('Existing server checkpoint differs from this notebook.')
    expected = {x['name'] for x in _ARC3_HELPERS['MODEL_MANIFEST'] if x['name'].endswith('.safetensors')}
    if set(report.get('staging', {}).get('verified_names', [])) != expected:
        raise RuntimeError('Existing model report does not confirm all original weight shards.')
    pid = report.get('server_pid')
    if not isinstance(pid, int) or pid <= 0:return False
    try:
        command = _ARC3Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        model = report['model_dir'].encode()
        if model not in command or not any(b'sglang' in arg for arg in command):return False
        endpoint = f"http://{_ARC3_HELPERS.get('SERVING_HOST','127.0.0.1')}:{_ARC3_HELPERS.get('SERVED_MODEL_PORT',8001)}"
        with _arc3_urlopen(endpoint+'/health', timeout=5) as response:
            if response.status != 200:return False
        with _arc3_urlopen(endpoint+'/v1/models', timeout=5) as response:
            models = _arc3_json.loads(response.read(65536))
    except (OSError, ValueError, KeyError):return False
    if not any(x.get('id') == _ARC3_HELPERS['CFG']['SERVED_NAME'] and x.get('max_model_len') == _ARC3_HELPERS['CFG']['CTX'] for x in models.get('data', [])):
        raise RuntimeError('Existing server exposes an unexpected model/context.')
    namespace = dict(_ARC3_HELPERS, MODEL_DIR=report['model_dir'],
        WHEELHOUSE_DIR=report['auxiliary_mounts']['wheelhouse'], DRAFT_VIEW=report['draft_view'],
        SGLANG=str(_ARC3Path(report['python_executable']).parent/'sglang'))
    exec(_ARC3_HELPERS['ORIGINAL_ARGUMENTS'], namespace)
    if namespace['args'] != report.get('serving_arguments'):
        raise RuntimeError('Existing server launch arguments differ from this notebook.')
    return True


def _arc3_shared_settings():
    """Notebook configuration values the harness process needs, as Python source."""
    values = {name: globals()[name] for name in ARC3_SHARED_SETTINGS}
    values.update(RUNTIME_PATHS=_ARC3_RUNTIME_PATHS, HARNESS_DIR=_ARC3Path(HARNESS_DIR),
                  NOTEBOOK_SUPPORT_DIR=_ARC3Path(NOTEBOOK_SUPPORT_DIR), ARC3_BUNDLE_NAME=ARC3_BUNDLE_NAME)
    lines = ['from pathlib import Path, PosixPath']
    lines += [f'{name} = {value!r}' for name, value in values.items()]
    return '\n'.join(lines)


def _arc3_prepare_colab():
    global _ARC3_BRIDGE, _ARC3_RUNTIME_PATHS, _ARC3_MODEL_REPORT, _ARC3_RUN_ROOT, _ARC3_EXPORT_COMPLETE
    _ARC3_EXPORT_COMPLETE = False
    helpers = _ARC3_HELPERS
    helpers.update(SETUP_ROOT=COLAB_MODEL_SETUP_ROOT, ARC3_DRIVE_DATASETS_ONLY=True,
        KAGGLE_SECRET_NAME=COLAB_KAGGLE_SECRET_NAME, KAGGLE_AUTH_MODE=COLAB_KAGGLE_AUTH_MODE,
        KAGGLE_USERNAME=COLAB_KAGGLE_USERNAME, KAGGLE_USERNAME_SECRET_NAME=COLAB_KAGGLE_USERNAME_SECRET_NAME,
        WEIGHT_DOWNLOAD_CONNECTIONS=COLAB_WEIGHT_DOWNLOAD_CONNECTIONS,
        MODEL_STARTUP_TIMEOUT_SECONDS=COLAB_MODEL_STARTUP_TIMEOUT_SECONDS,
        _ARC3_WHEELHOUSE_CACHE_ROOT=str(_ARC3Path(COLAB_CONFIG_PATH).parent/'cache'/'wheelhouse-v1'),
        REPORT_PATH=COLAB_RUNTIME_REPORT_PATH)
    if COLAB_MAIN_MODEL_HANDLE != 'dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1':
        raise RuntimeError('The hardcoded Drive manifest belongs to the original pinned checkpoint.')
    if helpers.get('SERVING_HOST','127.0.0.1') != '127.0.0.1':
        raise RuntimeError('The Colab model transport requires a loopback serving host.')
    old_bridge = globals().get('_ARC3_BRIDGE')
    if old_bridge is not None:old_bridge.close()
    run = _ARC3Path(COLAB_WORKING_ROOT)/('run-'+_arc3_uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    _ARC3_RUN_ROOT = run
    previous = _arc3_previous_report()
    setup = {'test':'ARC3 Colab runtime preparation', 'status':'PREPARING', 'run_root':str(run),
             'adapter':ARC3_ADAPTER_INFO}
    helpers['save_report'](setup)
    setup['drive_access'] = _arc3_check_drive_api()
    helpers['save_report'](setup)
    setup['auth_mode_used'] = 'not_required_drive_local_inputs'
    setup['input_source'] = 'Google Drive'
    setup['input_paths'] = {'harness': str(HARNESS_DIR)}
    setup['harness'] = dict(HARNESS_INFO)
    ARC3_ADAPTER_INFO['harness'] = {key: HARNESS_INFO.get(key) for key in ('name', 'version', 'tree_sha256', 'matches_manifest')}
    print('Harness:', HARNESS_INFO.get('version'), '|', str(HARNESS_DIR), flush=True)

    mounts = {'wheelhouse':COLAB_WHEELHOUSE_DIR}
    wheelhouse_path = _ARC3Path(COLAB_WHEELHOUSE_DIR)
    if not wheelhouse_path.is_dir() or not (wheelhouse_path/'wheels').is_dir() or not (wheelhouse_path/'requirements.lock').is_file():
        raise RuntimeError('Drive Pennyroyal wheelhouse layout is invalid: '+str(wheelhouse_path))
    setup['input_paths']['wheelhouse'] = str(wheelhouse_path)
    print('Drive input verified: Pennyroyal wheelhouse |',str(wheelhouse_path),flush=True)

    draft_resource = next(x for x in helpers['RESOURCES'] if x['name']=='draft_model')
    draft_path = _ARC3Path(COLAB_DRAFT_MODEL_DIR)
    draft_valid = False
    if draft_path.is_dir():
        draft_metadata = helpers['inspect_model_metadata'](draft_path,draft_resource)
        draft_valid = any(x.get('status')=='LAYOUT_OK' for x in draft_metadata.get('checkpoints',[]))
    if draft_valid:
        mounts['draft_model'] = str(draft_path)
        setup['auth_mode_used'] = 'not_required_existing_draft_model'
    else:
        session,mode = helpers['authenticated_session']()
        setup['auth_mode_used'] = mode
        helpers['_ARC3_RESOLVED_MOUNTS'] = dict(mounts)
        try:
            mounts.update(helpers['mount_model_dependencies'](session,[draft_resource],setup))
        finally:
            session.headers.pop('Authorization',None);session.auth=None;session.close()
    if not mounts.get('draft_model') or not _ARC3Path(mounts['draft_model']).is_dir():
        raise RuntimeError('Existing draft-model mount did not resolve.')
    setup['input_paths']['draft_model'] = str(mounts['draft_model'])
    print('Using the existing draft-model path:',str(mounts['draft_model']),flush=True)

    helpers['_ARC3_RESOLVED_MOUNTS'] = dict(mounts)
    source = COLAB_SOURCE_BUNDLE_DIR
    if not source or not helpers['validate_source_root'](source):
        raise RuntimeError('Drive source bundle layout invalid: '+str(source))
    setup['input_paths']['source_bundle'] = str(source)
    print('Drive input verified: source_bundle |',str(source),flush=True)

    competition = COLAB_COMPETITION_DIR
    if not competition or not helpers['competition_layout'](competition):
        raise RuntimeError('Drive competition data layout invalid: '+str(competition))
    setup['input_paths']['competition'] = str(competition)
    print('Drive input verified: competition |',str(competition),flush=True)
    helpers['save_report'](setup)
    helpers['_ARC3_RESOLVED_MOUNTS'] = dict(mounts)
    print('Checking the pinned Python 3.12 environment.',flush=True)
    python = str(_ARC3Path(COLAB_MODEL_SETUP_ROOT)/'venv/bin/python')
    if not _ARC3Path(python).is_file():
        helpers['prepare_python312'](setup, mounts['wheelhouse'])
        if setup.get('runtime_check', {}).get('status') != 'COMMAND_OK':
            raise RuntimeError('Pinned Python 3.12/CUDA setup failed; inspect colab_setup_report.json.')
    check = _arc3_subprocess.run([python, '-c',
        'import sys,importlib.metadata as m;assert sys.version_info[:2]==(3,12);'
        'assert m.version("torch")=="2.13.0+cu130";'
        'assert m.version("sglang")=="0.5.19+gd00d88efc8d6";import IPython'],
        capture_output=True, text=True, timeout=60)
    if check.returncode:raise RuntimeError('Pinned Python 3.12 environment check failed:\n'+check.stderr[-3000:])
    env = helpers['runtime_environment'](COLAB_MODEL_SETUP_ROOT, python)
    # Keep the offline ARC installation from changing the serving environment's existing pins.
    pins = _arc3_subprocess.run([python, '-c',
        'import importlib.metadata as m;print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions() if d.metadata.get("Name"))))'],
        capture_output=True, text=True, timeout=60, check=True)
    constraints = run/'existing-runtime-constraints.txt';constraints.write_text(pins.stdout)
    env['PIP_CONSTRAINT'] = str(constraints)
    _ARC3_RUNTIME_PATHS = {'wheelhouse':mounts['wheelhouse'], 'draft_model':mounts['draft_model'],
        'source_bundle':str(source), 'competition':str(competition), 'working':str(run/'working'),
        'bundle':str(run/'source'), 'harness':str(HARNESS_DIR), 'setup_root':COLAB_MODEL_SETUP_ROOT,
        'python':python, 'model':previous.get('model_dir',''), 'allow_harness_pypi':COLAB_ALLOW_HARNESS_PYPI,
        'model_settings':dict(COLAB_LLM_SETTINGS), 'serving_host':helpers.get('SERVING_HOST','127.0.0.1'),
        'serving_port':helpers.get('SERVED_MODEL_PORT',8001), 'serving_name':COLAB_LLM_SETTINGS['SERVED_NAME']}
    _ARC3_MODEL_REPORT = previous
    child_source = (_ARC3Path(NOTEBOOK_SUPPORT_DIR)/'colab_bridge_child.py').read_text(encoding='utf-8')
    _ARC3_BRIDGE = ARC3PythonBridge(python, child_source, run, env)
    _ARC3_BRIDGE.execute(_arc3_shared_settings())
    setup.update(status='HARNESS_READY', python_executable=python, paths=_ARC3_RUNTIME_PATHS,
        competition_data_note='Original competition offline wheels/environments.',
        drive_results_dir=COLAB_DRIVE_RESULTS_DIR)
    helpers['save_report'](setup)
    _arc3_record('HARNESS_READY')
    print('Colab runtime ready. Working directory:', _ARC3_RUNTIME_PATHS['working'], flush=True)


def _arc3_ensure_model_server():
    global _ARC3_MODEL_REPORT
    report = _arc3_previous_report()
    if COLAB_REUSE_MODEL_SERVER and _arc3_valid_server(report):
        print('Running model server verified; not reloading.', flush=True)
    else:
        with _arc3_socket.socket() as probe:
            port = _ARC3_HELPERS.get('SERVED_MODEL_PORT',8001)
            if probe.connect_ex((_ARC3_HELPERS.get('SERVING_HOST','127.0.0.1'), port)) == 0:
                raise RuntimeError(f'Port {port} is occupied by a server that cannot be reused. Keep the existing runtime or select a fresh runtime.')
        if not COLAB_LOAD_MODEL_IF_NEEDED:
            raise RuntimeError('No ready model server. Enable COLAB_LOAD_MODEL_IF_NEEDED or run the existing model-load notebook in this runtime.')
        helpers = _ARC3_HELPERS
        helpers['REPORT_PATH'] = COLAB_MODEL_REPORT_PATH
        report = helpers['run_model_load']()
        if report.get('status') != 'MODEL_HEALTHY':
            raise RuntimeError('Model startup failed: '+str(report.get('error_code', report.get('status'))))
        if not _arc3_valid_server(report):raise RuntimeError('Started model server did not pass reuse validation.')
    _ARC3_MODEL_REPORT = report
    _ARC3_RUNTIME_PATHS.update(model=report['model_dir'], draft_model=report['auxiliary_mounts']['draft_model'],
        wheelhouse=report['auxiliary_mounts']['wheelhouse'])
    _ARC3_BRIDGE.execute('RUNTIME_PATHS = '+repr(_ARC3_RUNTIME_PATHS)+'\nMODEL_DIR = '+repr(report['model_dir'])+
        '\nDRAFT_MODEL_DIR = '+repr(report['auxiliary_mounts']['draft_model'])+
        '\nWHEELHOUSE_DIR = '+repr(report['auxiliary_mounts']['wheelhouse']))
    working = _ARC3Path(_ARC3_RUNTIME_PATHS['working']);working.mkdir(parents=True, exist_ok=True)
    (working/'model_load_report.json').write_text(_arc3_json.dumps(report, ensure_ascii=False, indent=2))
    source_log = _ARC3Path(report.get('server_log_path',''))
    destination = working/'serve.log'
    if source_log.is_file() and not destination.exists():destination.symlink_to(source_log)


_ARC3_PERSIST_FILES = ('experiment_manifest.json', 'run_summary.json', 'benchmark.json', 'summary.txt',
                       'model_load_report.json', 'diagnostics.html', 'submission.parquet', 'colab_run_report.json',
                       'requests.zip')


def _arc3_persist_outputs(strict=False):
    """Copy the run outputs (or a partial bundle after a failure) to the Drive results folder."""
    root=globals().get('_ARC3_RUN_ROOT')
    if root is None:return None
    root=_ARC3Path(root);working=root/'working'
    destination=_ARC3Path(COLAB_DRIVE_RESULTS_DIR)/root.name
    try:
        destination.mkdir(parents=True,exist_ok=True)
        for source in (root/'notebook_console.log',_ARC3Path(COLAB_RUNTIME_REPORT_PATH)):
            if source.is_file():_arc3_shutil.copy2(source,destination/source.name)
        if working.is_dir():
            for name in _ARC3_PERSIST_FILES:
                source=working/name
                if source.is_file():_arc3_shutil.copy2(source,destination/name)
            bundles=[working/ARC3_BUNDLE_NAME] if (working/ARC3_BUNDLE_NAME).is_file() else []
            if not bundles:
                import zipfile
                partial=root/'partial_run_bundle.zip'
                with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                    for folder in ('observations','transcripts','artifacts','server_recording'):
                        for source in sorted((working/folder).rglob('*')):
                            if source.is_file():archive.write(source,str(source.relative_to(working)))
                    for name in _ARC3_PERSIST_FILES+('serve.log','git_status.txt'):
                        source=working/name
                        if source.is_file():archive.write(source,name)
                bundles=[partial]
                if not (working/'requests.zip').is_file():
                    sources=sorted(p for p in working.rglob('*requests.jsonl') if p.is_file())
                    if sources:
                        with zipfile.ZipFile(root/'requests.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                            for source in sources:archive.write(source,str(source.relative_to(working)))
                        bundles.append(root/'requests.zip')
            for source in bundles:_arc3_shutil.copy2(source,destination/source.name)
        print('Drive results folder:',str(destination),flush=True)
        return str(destination)
    except Exception as error:
        print('DRIVE_RESULTS_SAVE_FAILED:',type(error).__name__, '| local outputs:',str(root),flush=True)
        if strict:raise
        return None


def _arc3_download_bundle():
    """Add the Colab-side reports to the run bundle and copy everything to Drive."""
    path = _ARC3Path(_ARC3_RUNTIME_PATHS['working'])/ARC3_BUNDLE_NAME
    if not path.is_file():raise RuntimeError('Run bundle was not produced.')
    _arc3_record('RUN_COMPLETE')
    import zipfile
    with zipfile.ZipFile(path, 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        names = set(archive.namelist())
        for source,name in ((path.parent/'model_load_report.json','model_load_report.json'),
                            (path.parent/'colab_run_report.json','colab_run_report.json'),
                            (path.parent.parent/'notebook_console.log','notebook_console.log'),
                            (_ARC3Path(COLAB_RUNTIME_REPORT_PATH),'colab_setup_report.json')):
            if source.is_file() and name not in names:archive.write(source,name)
    _arc3_persist_outputs(strict=True)


def _arc3_record(status,cell=None,error=None,**extra):
    root=globals().get('_ARC3_RUN_ROOT')
    if root is None:return
    working=_ARC3Path(root)/'working';working.mkdir(parents=True,exist_ok=True)
    path=working/'colab_run_report.json'
    try:report=_arc3_json.loads(path.read_text()) if path.is_file() else {}
    except (ValueError,OSError):report={}
    import time
    report.update(status=status,run_root=str(root),adapter=ARC3_ADAPTER_INFO,
        harness_version=HARNESS_INFO.get('version'),harness_tree_sha256=HARNESS_INFO.get('tree_sha256'),
        run_stage=RUN_STAGE,pilot_game_prefixes=list(PILOT_GAME_PREFIXES),
        public_runtime_minutes=PUBLIC_RUNTIME_MINUTES,model_setup_separate_from_game_budget=True)
    event=dict(status=status,cell=cell,utc_epoch=time.time(),**extra)
    if error is not None:event.update(error_type=type(error).__name__,error_code=getattr(error,'code',type(error).__name__))
    report.setdefault('events',[]).append(event)
    if status=='MODEL_READY':report['model_health_confirmed']=True
    if status=='BENCHMARK_COMPLETED':report['benchmark_completed']=True
    if status=='RUN_COMPLETE':report['colab_end_to_end_validation']='RUN_COMPLETE'
    temporary=path.with_suffix('.json.tmp');temporary.write_text(_arc3_json.dumps(report,ensure_ascii=False,indent=2))
    temporary.replace(path)


def _arc3_start_model_for_experiment():
    import time
    start=time.monotonic();_arc3_record('MODEL_STARTING',cell='model')
    try:
        _arc3_ensure_model_server()
        _arc3_record('MODEL_READY',cell='model',model_setup_seconds=round(time.monotonic()-start,2))
    except BaseException as error:
        _arc3_record('MODEL_START_FAILED',cell='model',error=error)
        _arc3_persist_outputs()
        raise


def _arc3_execute(source,tag=None):
    """Run one notebook cell in the Python 3.12 harness process, with the Colab-side hooks."""
    global _ARC3_EXPORT_COMPLETE
    if tag=='model':_arc3_start_model_for_experiment()
    _arc3_record('CELL_STARTED',cell=tag)
    try:
        _ARC3_BRIDGE.execute(source)
        if tag=='provenance':
            _ARC3_BRIDGE.execute('_manifest["runtime_adapter"] = '+repr(ARC3_ADAPTER_INFO)+
                '\narc3nb.save_json(WORKING_DIR / "experiment_manifest.json", _manifest)')
        if tag=='export':
            _arc3_download_bundle()
            _ARC3_EXPORT_COMPLETE = True
        _arc3_record('BENCHMARK_COMPLETED' if tag=='run' else 'CELL_COMPLETED',cell=tag)
    except BaseException as error:
        _arc3_record('CELL_FAILED',cell=tag,error=error)
        if tag!='run':_arc3_persist_outputs()
        raise
    finally:
        if tag=='run':_arc3_persist_outputs()


# ---------------------------------------------------------------- cell forwarding
_ARC3_STEP_LINE = _arc3_re.compile(r'^\s*#\s*arc3-step:\s*([A-Za-z_-]+)\s*$')


def _arc3_forward_cell(lines):
    """IPython input transformer: send the cell to the harness process unless marked `parent`."""
    code = ''.join(lines)
    if not code.strip():
        return lines
    first = next((line for line in lines if line.strip()), '')
    match = _ARC3_STEP_LINE.match(first)
    tag = match.group(1).lower() if match else None
    if tag == 'parent':
        return lines
    return ['_arc3_execute(' + repr(code) + ', ' + repr(tag) + ')\n']


def _arc3_install_forwarding():
    shell = get_ipython()
    shell.input_transformers_cleanup[:] = [t for t in shell.input_transformers_cleanup
                                           if getattr(t, '__name__', '') != '_arc3_forward_cell']
    shell.input_transformers_cleanup.insert(0, _arc3_forward_cell)
    print('Later cells run in the Python 3.12 harness process '
          '(except cells whose first line is "# arc3-step: parent").', flush=True)


try:
    _arc3_prepare_colab()
except Exception as _setup_error:
    _report_path=_ARC3Path(COLAB_RUNTIME_REPORT_PATH)
    try:
        _failure=_arc3_json.loads(_report_path.read_text()) if _report_path.is_file() else {}
        _failure.update(status='SETUP_FAILED',error_type=type(_setup_error).__name__,
            error_code=getattr(_setup_error,'code','SEE_EXCEPTION'),
            http_status=getattr(_setup_error,'http_status',None),
            inference_settings_changed=False,model_loading_started=False)
        _ARC3_HELPERS['save_report'](_failure)
        _arc3_persist_outputs()
        print('Setup report:',str(_report_path),flush=True)
        print('Setup error:',getattr(_setup_error,'code',str(_setup_error)),
              '| HTTP:',getattr(_setup_error,'http_status',None),flush=True)
    except Exception:
        pass
    raise
_arc3_install_forwarding()
