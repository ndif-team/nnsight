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
      const color = darkMode ? 0xCECDC3 : 0x000000; // Example dark mode color
      const color2 = darkMode ? 0xCECDC3 : 0x000000; // Adjust color2 for dark mode as needed
      const backgroundColor = darkMode ? 0x100F0F : 0xFFFFFF; // Adjust background color for dark mode
      const size = darkMode ? 0.50 : 1.00; // Adjust size for dark mode as needed

      vantaEffect = VANTA.DOTS({
        el: ".fixed-background",
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
        size: size,
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
