from io import StringIO
from typing import Union

import requests
from rich.console import Console
from rich.table import Table

from . import CONFIG


class NdifStatus(dict):

    def __init__(self, response: dict):
        super().__init__(response)

        self._data = response
        self._table = self._table()

    @classmethod
    def request_status(cls):
        response = requests.get(f"http{'s' if {CONFIG.API.SSL} else ''}://{CONFIG.API.HOST}/status")
        response = response.json()
        return response

    def _color_state(self, state: str):
        if state == "RUNNING":
            return "green"
        elif state == "DEPLOYING" or state == "NOT_STARTED":
            return "yellow"
        else:
            return "red"

    def _color_type(self, type: str):
        if type == "Dedicated":
            return "green"
        elif type == "Pilot-Only":
            return "purple"
        elif "Scheduled":
            return "blue"

    def _service_status(self):
        if any([value['state'] == "RUNNING" for value in self._data.values()]):
            return "NDIF Service: Running 🟢"
        elif any([value['state'] == "DEPLOYING" or value['state'] == "NOT_STARTED" for value in self._data.values()]):
            return "NDIF Service: Redeploying 🟡"
        else:
            return "NDIF Service: Down 🔴"

    def _table(self):
        table = Table()
        table.add_column("Model Class")
        table.add_column("Repo ID", no_wrap=True)
        table.add_column("Revision")
        table.add_column("Type")
        table.add_column("Status")

        for key, value in self.items():
            table.add_row(
                value['model_class'], 
                value['repo_id'], 
                value['revision'], 
                f"[b {self._color_type(value['type'])}]{value['type']}[/]", 
                f"[b {self._color_state(str(value['state']))}]{str(value['state'])}[/]",
            )
        
        return table

    def __str__(self):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, stderr=False)
        console.print(self._service_status(), "\n", self._table)
        return buf.getvalue()

    def __repr__(self):
        return self.__str__()

def ndif_status(raw: bool = False) -> Union[dict, NdifStatus]:
    response = NdifStatus.request_status()

    if raw:
        return response
    else:
        formatted_response = {}
        for key, value in response['deployments'].items():
            if value['deployment_level'] == 'HOT' or value['deployment_level'] == 'WARM' or 'schedule' in value:
                model_class = value['model_key'].split('.')[3].split(':')[0]
                repo_id = value['repo_id']
                revision = value['model_key'].split('\"')[-2]

                state = value.get('application_state', "NOT DEPLOYED")
                if 'dedicated' in value:
                    type = 'Dedicated' if value['dedicated'] == True else 'Pilot-Only'
                elif 'schedule' in value:
                    type = 'Scheduled'

                formatted_response[repo_id] = {
                    'model_class': model_class,
                    'repo_id': repo_id,
                    'revision': revision,
                    'type': type,
                    'state': state,
                }

        return NdifStatus(formatted_response)


def is_model_running(repo_id: str, revision: str = "main") -> bool:
    response = NdifStatus.request_status()

    for key, value in response['deployments'].items():
        if value['repo_id'] == repo_id and value['revision'] == revision:
            return value.get('application_state', None) == "RUNNING"
    
    return False