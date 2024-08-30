:html_theme.sidebar_secondary.remove:
:sd_hide_title:

.. raw:: html

    <style>
        .accordion-header {
            margin: 0 !important;
        }
        /* Custom accordion styles */
        .custom-accordion-header {
            background-color: var(--pst-color-surface); /* Default state background color */
            color: var(--pst-color-text-base); /* Text color */
            border-bottom: var(--pst-color-border); /* Border color */
        }

        .custom-accordion-header: {
            background-color: var(--pst-color-surface); /* Default state background color */
            color: var(--pst-color-text-base); /* Text color */
            border-bottom: var(--pst-color-border); /* Border color */
        }

        .accordion {
            --bs-accordion-btn-icon: none;
            --bs-accordion-btn-active-icon: none;
        }

        .custom-accordion-header.collapsed {
            background-color: var(--pst-color-on-background); /* Collapsed state background color */
            color: var(--pst-color-text-base); /* Text color */
        }

        .custom-accordion-header:not(.collapsed) {
            background-color: var(--pst-color-surface); /* Active/Expanded state background color */
            color: var(--pst-color-text-base); /* Text color */
        }

        .custom-accordion-body {
            background-color: var(--pst-color-on-background); /* Body background color */
            border-color: var(--pst-color-border); /* Border color */
            color: var(--pst-color-text-base); /* Text color */
        }

        .sd-card {
            border-radius: 0 !important;
        }
    </style>

    
    <script>

        function autoFormatJsonString(jsonString) {
            // Parse the JSON string into an object
            let jsonObject = JSON.parse(jsonString);

            // Convert the object back into a string with indentation
            let prettyPrintedJson = JSON.stringify(jsonObject, null, 2);

            // Replace keys in the JSON string with styled spans
            prettyPrintedJson = prettyPrintedJson.replace(/"([^"]+)":/g, '<span style="background-color: lightgrey;">"$1":</span>');

            // Set the formatted JSON string as the innerHTML of the element
            document.getElementById('jsonContainer').innerHTML = `<pre>${prettyPrintedJson}</pre>`;
        }


        fetch("https://ndif.dev/ping")
            .then((response) => {
                if (response.status == 200) {
                    document.querySelectorAll('div.sd-card-body.status-container').forEach(el => {
                        el.style.backgroundColor = "#66800b";
                        el.querySelectorAll('p.sd-card-text').forEach(el => {
                            el.textContent = "All Systems Are Operational";
                        });
                    });


                    console.log('Ping success');

                    // Nested fetch to ndif.dev/stats
                    fetch("https://ndif.dev/stats")
                        .then((statsResponse) => {
                            if (statsResponse.status == 200) {
                                statsResponse.json().then((parsed) => {
                                    // Initialize an empty string to accumulate information
                                    let infoString = '';

                                    let index = 0;

                                    let modelSummary = {};

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
                                console.log('Stats error');
                            }
                        })
                        .catch((statsError) => {
                            console.log('Stats error');
                        });
                } else {
                    document.querySelectorAll('div.sd-card-body.status-container').forEach(el => {
                        el.style.backgroundColor = "#F84F31";
                        el.querySelectorAll('p.sd-card-text').forEach(el => {
                            el.textContent = "NDIF Is Unavailable";
                        });
                    });
                    console.log('Ping error');
                }
            })
            .catch((pingError) => {
                document.querySelectorAll('div.sd-card-body.status-container').forEach(el => {
                    el.style.backgroundColor = "#F84F31";
                    el.querySelectorAll('p.sd-card-text').forEach(el => {
                        el.textContent = "NDIF Is Unavailable";
                    });
                });
                console.error('Ping fetch failed:', pingError);
            });
    </script>


Status
======

.. card::
    :class-body: status-container
    :shadow: none
    
    All Systems Are Operational

.. card::
    :shadow: none
    
    The library can be used to run local models without requiring a key. However, running experiments on remote models requires a free server API key. To obtain a key, register for an `NDIF account <https://login.ndif.us>`_ which allows you to manage and generate keys.

.. raw:: html

    <div class="accordion accordion-flush" id="accordionHook">
    </div>
