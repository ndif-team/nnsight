:html_theme.sidebar_secondary.remove:
:sd_hide_title:

.. raw:: html

    <link rel="stylesheet" href="../_static/css/status.css">
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
                    Array.from(document.getElementsByClassName("status-container")).forEach((elm) => {
                        elm.style.backgroundColor = "#23C552";
                        Array.from(elm.getElementsByClassName('sd-card-text')).forEach((text) => {
                            text.textContent = "All Systems Are Operational";
                            text.style.color = "#FFFFFF";
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
                                        
                                        infoString += `<div class="accordion-item"> <h2 class="accordion-header" id="${headingId}"> <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}"> (${summaryItem.number_of_copies}x) ${key} </button> </h2> <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="${headingId}" data-bs-parent="#accordionExample"> <div class="accordion-body">${huggingFaceLink}</i> <pre>${prettyPrintedJson}</pre></div> </div> </div>`;

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
                    Array.from(document.getElementsByClassName("status-container")).forEach((elm) => {
                        elm.style.backgroundColor = "#F84F31";
                        Array.from(elm.getElementsByClassName('sd-card-text')).forEach((text) => {
                            text.textContent = "NDIF Is Unavailable";
                            text.style.color = "#FFFFFF";
                        });
                    });
                    console.log('Ping error');
                }
            })
            .catch((pingError) => {
                Array.from(document.getElementsByClassName("status-container")).forEach((elm) => {
                    elm.style.backgroundColor = "#F84F31";
                    Array.from(elm.getElementsByClassName('sd-card-text')).forEach((text) => {
                        text.textContent = "NDIF Is Unavailable";
                        text.style.color = "#FFFFFF";
                    });
                });
                console.error('Ping fetch failed:', pingError);
            });
    </script>


Status
======



.. toctree::
   :maxdepth: 1
   :hidden:

.. card::
    :class-card: status-container
    
    All Systems Are Operational

.. raw:: html

    <div class="accordion accordion-flush" id="accordionHook">
    </div>