"""Python 3.12 harness process for Colab: runs forwarded notebook cells in an IPython shell
and streams output/display events back to the Colab kernel as JSON lines."""
import asyncio
import io
import json
import sys
import threading
import traceback
from IPython.core.interactiveshell import InteractiveShell

_input = sys.stdin
_output = sys.stdout
_lock = threading.Lock()


def emit(kind, **fields):
    with _lock:
        _output.write(json.dumps(dict(arc3_bridge_event=kind, **fields), ensure_ascii=False) + '\n')
        _output.flush()


class Stream(io.TextIOBase):
    def __init__(self, name):self.name = name
    def write(self, text):
        if text:emit('stream', name=self.name, text=text)
        return len(text)
    def flush(self):_output.flush()
    def isatty(self):return False
    def fileno(self):return _output.fileno()


sys.stdout = Stream('stdout')
sys.stderr = Stream('stderr')
shell = InteractiveShell.instance()
shell.colors = 'nocolor'
shell.autoawait = True
def publish(data, metadata=None, **kwargs):
    emit('display', data=data, metadata=metadata or {})
shell.display_pub.publish = publish
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
emit('ready', python=sys.version.split()[0])
for line in _input:
    command = json.loads(line)
    if command.get('action') == 'close':break
    try:
        transformed = shell.transform_cell(command['code'])
        result = loop.run_until_complete(shell.run_cell_async(command['code'], store_history=False,
            transformed_cell=transformed, preprocessing_exc_tuple=None))
        error = result.error_before_exec or result.error_in_exec
        emit('done', success=result.success, error_type=type(error).__name__ if error else None)
    except BaseException as error:
        traceback.print_exc()
        emit('done', success=False, error_type=type(error).__name__)
loop.close()
