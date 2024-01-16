:html_theme.sidebar_secondary.remove:
:sd_hide_title:

.. raw:: html

    <script>
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

                                    // Iterate through the JSON dictionary and append information
                                    Object.keys(parsed).forEach((key) => {
                                        const value = parsed[key];
                                        infoString += `Repo ID: <a href="https://huggingface.co/${value.repo_id}" target="_blank">${value.repo_id}</a><br>Config String: ${value.config_json_string}<br><br>`;
                                    });

                                    // Display the accumulated string
                                    Array.from(document.getElementsByClassName("status-message")).forEach((elm) => {
                                        Array.from(elm.getElementsByClassName('sd-card-text')).forEach((text) => {
                                            text.innerHTML = infoString;
                                        });
                                    });

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

.. card::
    :class-card: status-message
    
    Loading...