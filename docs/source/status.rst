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

        .accordion-item {
            border: 1px solid var(--pst-color-border);
            margin-bottom: 0.5rem;
            border-radius: 4px;
        }

        .accordion {
            --bs-accordion-btn-icon: none;
            --bs-accordion-btn-active-icon: none;
        }

        .custom-accordion-header.collapsed {
            background-color: var(--pst-color-surface);
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
            background-color: var(--pst-color-surface);
            /* Body background color */
            border-color: var(--pst-color-border);
            /* Border color */
            color: var(--pst-color-text-base);
            /* Text color */
            padding: 1rem;
        }

        .sd-card {
            border-radius: 0 !important;
        }

        /* Add styles for status text */
        .status-container p.sd-card-text {
            font-weight: 600;
            margin: 0;
            padding: 0.5rem 1rem;
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
        let error_color = "#7e0000"  // Red for FAILED/UNHEALTHY
        let success_color = "#66800b"  // Green for RUNNING
        let warning_color = "#7d7106"  // Yellow for other states

        function getStatusColor(status) {
            switch(status) {
                case "RUNNING":
                    return success_color;
                case "DEPLOY_FAILED":
                case "UNHEALTHY":
                    return error_color;
                default:
                    return warning_color;
            }
        }

        function formatTimeRemaining(endTime) {
            const now = new Date();
            const end = new Date(endTime);
            
            const diff = end - now;
            
            if (diff < 0) return "Ended";
            
            const hours = Math.floor(diff / (1000 * 60 * 60));
            const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
            return `${hours}h ${minutes}m remaining`;
        }

        function formatSchedule(schedule) {
            if (!schedule) return "No schedule";
            
            const now = new Date();
            const start = new Date(schedule.start_time);
            const end = new Date(schedule.end_time);
            
            if (now > end) return "Ended";
            if (now >= start && now <= end) {
                return formatTimeRemaining(schedule.end_time);
            }
            if (now < start) {
                const startStr = start.toLocaleString();
                const duration = Math.round((end - start) / (1000 * 60 * 60));
                return `Starts ${startStr} (${duration}h duration)`;
            }
        }

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

        function updateScheduleDisplay() {
            document.querySelectorAll('.schedule-info').forEach(el => {
                const schedule = JSON.parse(el.dataset.schedule);
                el.textContent = formatSchedule(schedule);
            });
        }

        function startScheduleTimer() {
            // Update immediately
            updateScheduleDisplay();
            // Then update every minute
            setInterval(updateScheduleDisplay, 60000);
        }

        document.addEventListener('DOMContentLoaded', function() {
            loading(true);

            update("Fetching NDIF status...", warning_color);

            fetch(ndif_url + "/ping")
                .then((response) => {
                    if (response.status == 200) {
                        update("NDIF is up. Fetching model status...", warning_color);

                        console.log('Ping success');
                        fetch(ndif_url + "/status")
                            .then((statsResponse) => {
                                loading(false);

                                if (statsResponse.status == 200) {
                                    statsResponse.json().then((response) => {
                                        console.log('Parsed response:', response);
                                        
                                        let infoString = '';
                                        let index = 0;

                                        if (!response.deployments || Object.keys(response.deployments).length === 0) {
                                            update("NDIF is up but there are no models deployed. Seems unintentional.", error_color);
                                            return;
                                        }

                                        update("NDIF is operational.", success_color);

                                        // Add calendar link if available
                                        const calendarLink = response.calendar_id ? 
                                            `<a href="https://calendar.google.com/calendar/embed?src=${encodeURIComponent(response.calendar_id)}" target="_blank" style="display: block; margin-bottom: 1rem; text-decoration: none;">View Deployment Calendar ↗</a>` : '';

                                        Object.entries(response.deployments).forEach(([key, value]) => {
                                            var headingId = 'heading' + (index + 1);
                                            var collapseId = 'collapse' + (index + 1);

                                            const configJsonString = value.config_json_string;
                                            const status = value.status;
                                            const schedule = value.schedule;

                                            let jsonObject = JSON.parse(configJsonString);
                                            let prettyPrintedJson = '';
                                            if (Object.keys(jsonObject).length > 0) {
                                                prettyPrintedJson = JSON.stringify(jsonObject, null, 4);
                                                prettyPrintedJson = prettyPrintedJson.replace(/"([^"]+)":/g, '"<b>$1</b>":');
                                            }
                                            let huggingFaceLink = `<a href="http://huggingface.co/${value.title}" target="_blank">HuggingFace Model Repository ↗</a>`;

                                            const statusColor = getStatusColor(status);
                                            const statusBadge = `<span style="background-color: ${statusColor}; color: white; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">${status}</span>`;
                                            
                                            // Create schedule info with data attribute for updates
                                            const scheduleInfo = schedule ? 
                                                `<span class="schedule-info" data-schedule='${JSON.stringify(schedule)}' style="background-color: #f8f9fa; color: #495057; padding: 2px 8px; border-radius: 4px; margin-left: 8px; border: 1px solid #dee2e6;">${formatSchedule(schedule)}</span>` : '';

                                            infoString += `<div class="accordion-item">
                                                    <h2 class="accordion-header" id="${headingId}">
                                                        <button class="accordion-button custom-accordion-header collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}" style="display: flex; justify-content: space-between; align-items: center; width: 100%; text-align: left;">
                                                            <div style="display: flex; align-items: center; flex: 1;">
                                                                ${statusBadge}<span style="font-weight: 600;">${value.title}</span>
                                                            </div>
                                                            <div style="margin-left: auto;">
                                                                ${scheduleInfo}
                                                            </div>
                                                        </button>
                                                    </h2>
                                                    <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="${headingId}" data-bs-parent="#accordionExample">
                                                        <div class="accordion-body custom-accordion-body">
                                                            ${huggingFaceLink}
                                                            ${prettyPrintedJson ? `<pre>${prettyPrintedJson}</pre>` : ''}
                                                        </div>
                                                    </div>
                                                </div>`;

                                            index++;
                                        });

                                        var elm = document.getElementById("accordionHook");
                                        elm.innerHTML = calendarLink + infoString;

                                        // Start the schedule update timer
                                        startScheduleTimer();

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