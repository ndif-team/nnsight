:html_theme.sidebar_secondary.remove:
:sd_hide_title:

.. raw:: html

    <style>
        .accordion-header {
            margin: 0 !important;
        }

        /* Custom accordion styles */
        .custom-accordion-header {
            background-color: var(--pst-color-surface);
            /* Default state background color */
            color: var(--pst-color-text-base);
            /* Text color */
            border-bottom: var(--pst-color-border);
            /* Border color */
        }

        .custom-accordion-header {
            background-color: var(--pst-color-surface);
            /* Default state background color */
            color: var(--pst-color-text-base);
            /* Text color */
            border-bottom: var(--pst-color-border);
            /* Border color */
        }

        .accordion {
            --bs-accordion-btn-icon: none;
            --bs-accordion-btn-active-icon: none;
        }

        .custom-accordion-header.collapsed {
            background-color: var(--pst-color-on-background);
            /* Collapsed state background color */
            color: var(--pst-color-text-base);
            /* Text color */
        }

        .custom-accordion-header:not(.collapsed) {
            background-color: var(--pst-color-surface);
            /* Active/Expanded state background color */
            color: var(--pst-color-text-base);
            /* Text color */
        }

        .custom-accordion-body {
            background-color: var(--pst-color-on-background);
            /* Body background color */
            border-color: var(--pst-color-border);
            /* Border color */
            color: var(--pst-color-text-base);
            /* Text color */
        }

        .sd-card {
            border-radius: 0 !important;
        }

        #loader {
            width: 120px;
            height: 120px;
            display: inline-block;
            position: relative;
        }
        #loader::after,
        #loader::before {
            content: '';  
            box-sizing: border-box;
            width:120px;
            height: 120px;
            border-radius: 50%;
            background: #FFF;
            position: absolute;
            left: 0;
            top: 0;
            animation: animloader 2s linear infinite;
        }
        #loader::after {
            animation-delay: 1s;
        }
        
        @keyframes animloader {
            0% {
                transform: scale(0);
                opacity: 1;
            }
            100% {
                transform: scale(1);
                opacity: 0;
            }
        }

    </style>


    <script>

        let ndif_url = "https://ndif.dev"
        let error_color = "#7e0000"
        let success_color = "#66800b"
        let warning_color = "#7d7106"

        function autoFormatJsonString(jsonString) {
            // Parse the JSON string into an object
            let jsonObject = JSON.parse(jsonString);

            // Convert the object back into a string with indentation
            let prettyPrintedJson = JSON.stringify(jsonObject, null, 2);

            // Replace keys in the JSON string with styled spans
            prettyPrintedJson = prettyPrintedJson.replace(/"([^"]+)":/g, '<span style="background-color: lightgrey;">"$1":</span>');

            // Set the formatted JSON string as the innerHTML of the element
            document.getElementById('jsonContainer').innerHTML = `<pre>${prettyPrintedJson}</pre>`;
        };

        function update(message, color) {
            document.querySelectorAll('div.sd-card-body.status-container').forEach(el => {
                el.style.backgroundColor = color;
                el.querySelectorAll('p.sd-card-text').forEach(el => {
                    el.textContent = message;
                });
            });
        }

        function loading(flag) {
            document.getElementById("loader").style.display = flag ? "block" : "none";
        }

        document.addEventListener('DOMContentLoaded', function() {

            loading(true);

            update("Fetching NDIF status...", warning_color);

            fetch(ndif_url + "/ping")

                .then((response) => {
                    if (response.status == 200) {

                        update("NDIF is up. Fetching model status...", warning_color);

                        console.log('Ping success');
                        // Nested fetch to ndif.dev/stats
                        fetch(ndif_url + "/stats")
                            .then((statsResponse) => {

                                loading(false);

                                if (statsResponse.status == 200) {
                                    statsResponse.json().then((parsed) => {
                                        // Initialize an empty string to accumulate information
                                        let infoString = '';

                                        let index = 0;

                                        let modelSummary = {};

                                        if (parsed.length === 0) {

                                            update("NDIF is up but there are no models deployed. Seems unintentional.", error_color);

                                            return
                                        }


                                        update("NDIF is operational.", success_color);

                                        Object.values(parsed).forEach((value) => {
                                            // Create a unique key for each model-config combination
                                            let modelConfigKey = `${value.repo_id}`;

                                            // Check if this model-config combination already exists in the summary
                                            if (modelSummary[modelConfigKey]) {
                                                // Increment the count if it does
                                                modelSummary[modelConfigKey].number_of_copies += 1;
                                            } else {
                                                // Otherwise, add a new entry
                                                modelSummary[modelConfigKey] = {
                                                    number_of_copies: 1,
                                                    config_string: value.config_json_string
                                                };
                                            }
                                        });

                                        // Now modelSummary contains the consolidated information
                                        console.log(modelSummary);

                                        // Iterate through the JSON dictionary and append information
                                        // Iterate through the modelSummary dictionary and append information
                                        Object.keys(modelSummary).forEach((key) => {
                                            var headingId = 'heading' + (index + 1);
                                            var collapseId = 'collapse' + (index + 1);

                                            const summaryItem = modelSummary[key];
                                            const configJsonString = summaryItem.config_string;

                                            let jsonObject = JSON.parse(configJsonString);

                                            // Convert the object back into a string with indentation
                                            let prettyPrintedJson = JSON.stringify(jsonObject, null, 4);

                                            prettyPrintedJson = prettyPrintedJson.replace(/"([^"]+)":/g, '"<b>$1</b>":');
                                            let huggingFaceLink = `<a href="http://huggingface.co/${key}" target="_blank">HuggingFace Model Repository ↗</a>`;

                                            infoString += `<div class="accordion-item">
                                                    <h2 class="accordion-header" id="${headingId}">
                                                        <button class="accordion-button custom-accordion-header collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}">
                                                            (${summaryItem.number_of_copies}x) ${key}
                                                        </button>
                                                    </h2>
                                                    <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="${headingId}" data-bs-parent="#accordionExample">
                                                        <div class="accordion-body custom-accordion-body">${huggingFaceLink}<pre>${prettyPrintedJson}</pre></div>
                                                    </div>
                                                </div>`;


                                            index++;
                                        });

                                        var elm = document.getElementById("accordionHook");

                                        elm.innerHTML = infoString;


                                        console.log('Stats success');
                                    }).catch((jsonError) => {
                                        console.log('JSON parsing error:', jsonError);
                                    });
                                } else {
                                    update("Unable to get NDIF status.", error_color);

                                }
                            })
                            .catch((statsError) => {
                                update("Unable to get NDIF status.", error_color);
                                loading(false);

                                console.log('Stats error', statsError);
                            });
                    } else {
                        update("NDIF is unavailable", error_color);
                        loading(false);
                        console.log('Ping error');
                    }
                })
                .catch((pingError) => {
                    update("NDIF is unavailable", error_color);
                    loading(false);
                    console.error('Ping fetch failed:', pingError);
                });

        }, false);
    </script>


Status
======

.. card::
    :class-body: status-container
    :shadow: none

    Getting Status

.. card::
    :shadow: none
    
    The library can be used to run local models without requiring a key. However, running experiments on remote models requires a free server API key. To obtain a key, register for an `NDIF account <https://login.ndif.us>`_ which allows you to manage and generate keys.
    For information on API key configuration and remote system limits, please refer to our `Remote Execution Tutorial <https://nnsight.net/notebooks/features/remote_execution/>`_.

    We currently have engineers on call Monday to Friday from 9 AM to 5 PM ET to assist with any connectivity issues for our remote models. Please reach out to us on `Discord <https://discord.com/invite/6uFJmCSwW7>`_ or at mailto:info@ndif.us.

.. raw:: html

    <div style="
        width:100%;
        display: flex;
        justify-content: center;
        ">
        <div id="loader"></div>
    </div>
    


    <div class="accordion accordion-flush" id="accordionHook">
    </div>