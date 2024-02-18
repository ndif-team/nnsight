require.config({
  paths: {
    'three': 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js?MEH=BLAH',
    'VANTA': 'https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.dots.min.js?MEH=BLAH'
  },
  shim: {
    'VANTA': {
      deps: ['three'] 
    }
  }
});

var vantaEffect;

function initVanta() {
  require(['three'], function (THREE) {
    window.THREE = THREE; // Make THREE globally accessible

    require(['VANTA'], function () {
      if (vantaEffect) {
        vantaEffect.destroy();
      }
      
      const darkMode = document.documentElement.dataset.theme === 'dark';
      const color = darkMode ? 0xCECDC3 : 0x100F0F; // Example dark mode color
      const color2 = darkMode ? 0xCECDC3 : 0x100F0F; // Adjust color2 for dark mode as needed
      const backgroundColor = darkMode ? 0x100F0F : 0xFFFCF0; // Adjust background color for dark mode

      vantaEffect = VANTA.DOTS({
        el: "#canvas",
        mouseControls: false,
        touchControls: false,
        gyroControls: false,
        minHeight: 200.00,
        minWidth: 200.00,
        scale: 1.00,
        scaleMobile: 1.00,
        color: color,
        color2: color2,
        backgroundColor: backgroundColor,
        size: 0.50,
        spacing: 10.00,
        showLines: false
      });
    });
  });
}

// Initial call
initVanta();

// Setup a mutation observer to detect theme changes
var observer = new MutationObserver(function(mutations) {
  mutations.forEach(function(mutation) {
    if (mutation.attributeName === "data-theme") {
      initVanta(); // Reinitialize Vanta with the new theme settings
    }
  });
});

// Start observing the document element for attribute changes
observer.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});



document.addEventListener('DOMContentLoaded', (event) => {
  var listItems = document.querySelectorAll('#list-tab .list-group-item');
  listItems.forEach(function (listItem) {
    listItem.addEventListener('click', function (e) {
      e.preventDefault();
      var tabPaneId = this.getAttribute('href');

      // Remove active class from all list items
      listItems.forEach(function (item) {
        item.classList.remove('active');
      });

      // Add active class to the clicked list item
      this.classList.add('active');

      // Hide all tab panes
      var tabPanes = document.querySelectorAll('.tab-content .tab-pane');
      tabPanes.forEach(function (tabPane) {
        tabPane.classList.remove('show');
        tabPane.classList.remove('active');
      });

      // Show the clicked tab pane
      var activeTabPane = document.querySelector(tabPaneId);
      activeTabPane.classList.add('active');

      // We need to manually add 'show' class after a slight delay to trigger fade effect
      setTimeout(function () {
        activeTabPane.classList.add('show');
      }, 150); // Adjust delay as needed
    });
  });
});

document.addEventListener("DOMContentLoaded", function() {
  fetch("https://ndif.dev/ping")
    .then(response => response.json())
    .then(data => {
      const statusElement = document.querySelector(".ndif");
      const status = (data === "pong") ? "Available" : "Unavailable";
      if (statusElement) {
        statusElement.setAttribute('data-bs-original-title', `Status: ${status}`);
      }
    })
    .catch(error => console.error('Error fetching NDIF status:', error));

    const data = document.querySelector("p.copyright");
    data.innerHTML = "(c) 2024 <a href='https://ndif.us/'>NDIF</a>"

});
