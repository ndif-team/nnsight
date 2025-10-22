from io import StringIO

import requests
from rich.console import Console
from rich.table import Table

from . import CONFIG


class NdifStatus(dict):

    def __init__(self, response: dict):
        super().__init__(response)

        self._data = response
        self._table = self._table()


    def _color_state(self, state: str):
        if state == "RUNNING":
            return "green"
        elif state == "DEPLOYING" or state == "NOT_STARTED":
            return "yellow"
        else:
            return "red"

    def _service_status(self):
        if any([value['state'] == "RUNNING" for value in self._data.values()]):
            return "NDIF Service: Running 🟢"
        elif any([value['state'] == "DEPLOYING" or value['state'] == "NOT_STARTED" for value in self._data.values()]):
            return "NDIF Service: Redeploying 🟡"
        else:
            return "NDIF Service: Down 🔴"

    def _table(self):
        table = Table("Model Class", "Repo ID", "Revision", "Dedicated", "Status")
        for key, value in self.items():
            table.add_row(value['model_class'], value['repo_id'], value['revision'], str(value['dedicated']), f"[b {self._color_state(str(value['state']))}]{str(value['state'])}[/]")
        return table

    def __str__(self):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, stderr=False)
        console.print(self._service_status(), "\n", self._table)
        return buf.getvalue()

    def __repr__(self):
        return self.__str__()

def ndif_status(raw: bool = False):
    response = requests.get(f"http{'s' if {CONFIG.API.SSL} else ''}://{CONFIG.API.HOST}/status")
    response = response.json()

    if raw:
        return response
    else:
        formatted_response = {}
        for key, value in response['deployments'].items():
            if value['deployment_level'] == 'HOT' or value['deployment_level'] == 'WARM':
                model_class = value['model_key'].split('.')[3].split(':')[0]
                repo_id = value['repo_id']
                revision = value['model_key'].split('\"')[-2]
                # state = f"[b {color_state(str(value['application_state']))}]{str(value['application_state'])}[/]"
                state = value['application_state']
                dedicated = str(value['dedicated'])
                formatted_response[repo_id] = {
                    'model_class': model_class,
                    'repo_id': repo_id,
                    'revision': revision,
                    'dedicated': dedicated,
                    'state': state
                }

        return NdifStatus(formatted_response)


def is_model_running(repo_id: str):
    response = requests.get(f"http{'s' if {CONFIG.API.SSL} else ''}://{CONFIG.API.HOST}/status")
    response = response.json()

    for key, value in response['deployments'].items():
        if value['repo_id'] == repo_id:
            return value.get('application_state', None) == "RUNNING"
    
    return False