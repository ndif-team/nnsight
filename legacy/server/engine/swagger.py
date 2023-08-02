from engine.models.result import Result
from engine.models.submit import Request
from openapi_schema_pydantic.v3.v3_0_3 import OpenAPI
from openapi_schema_pydantic.v3.v3_0_3.util import (PydanticSchema,
                                          construct_open_api_with_schema_class)


def construct_base_open_api(config):

    return OpenAPI.parse_obj({
        "servers": [{
            "url": "http://nagoya.research.khoury.northeastern.edu"
        }],
        "info": {
            "version": "1.0.0",
            "title": "NDIF API",
            "description": "An API for free and open access to large language models"},
        "paths": {
            f"/{config['SUBMIT_EP']}": {
                "post": {
                    "summary": "Submits job for processing",
                    "requestBody": {"content": {"application/json": {
                        "schema": PydanticSchema(schema_class=Request)
                    }}},
                    "responses": {"200": {
                        "description": "Success",
                        "content": {"application/json": {
                            "schema": PydanticSchema(schema_class=Result)
                        }},
                    }},
                }
            },
            f"/{config['RETRIEVE_EP']}/{{id}}": {
                "get": {
                    "summary": "Retrieves job status/results",
                    "parameters": [{
                        "name": "job_id",
                        "in" : "path",
                        "description": "Job ID",
                        "required": True,
                        "schema": {
                            "type": "integer"
                        }
                    }],
                    "responses": {"200": {
                        "description": "Success",
                        "content": {"application/json": {
                            "schema": PydanticSchema(schema_class=Result)
                        }},
                    }},
                },
            },
        },
    })

def generate(config):

    open_api = construct_base_open_api(config)
    open_api = construct_open_api_with_schema_class(open_api)

    return open_api.json(by_alias=True, exclude_none=True, indent=2)