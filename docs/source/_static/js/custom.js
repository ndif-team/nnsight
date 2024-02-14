require.config({
    paths: {
        'three': 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js?MEH=BLAH',
        'VANTA': 'https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.dots.min.js?MEH=BLAH'
    },
    shim: {
        'VANTA': {
            deps: ['three'] // Removed the exports line
        }
    }
});

// require(['three', 'VANTA'], function(THREE) {
//     // Assuming VANTA attaches itself to a global variable, directly use it here
//     // Check the correct global variable name (VANTA, _vantaEffect, etc.)
//     if (window.VANTA && window.VANTA.WAVES) {
//         window.VANTA.WAVES({
//             el: "#hero"
//         });
//     } else {
//         console.error('VANTA.WAVES is not available.');
//     }
// });

require(['three'], function(THREE) {
    window.THREE = THREE; // Make THREE globally accessible

    require(['VANTA'], function() {
        if (VANTA.DOTS) {
            VANTA.DOTS({
                el: "#canvas",
                mouseControls: false,
                touchControls: false,
                gyroControls: false,
                minHeight: 200.00,
                minWidth: 200.00,
                scale: 1.00,
                scaleMobile: 1.00,
                color: 0xb3a094,
                color2: 0xffffff,
                backgroundColor: 0x100F0F,
                size: 0.50,
                // spacing: 68.00,
                spacing: 10.00,
                showLines: false
            })
        }
    });
});


document.addEventListener('DOMContentLoaded', (event) => {
    var listItems = document.querySelectorAll('#list-tab .list-group-item');
    listItems.forEach(function(listItem) {
      listItem.addEventListener('click', function(e) {
        e.preventDefault();
        var tabPaneId = this.getAttribute('href');
        
        // Remove active class from all list items
        listItems.forEach(function(item) {
          item.classList.remove('active');
        });
        
        // Add active class to the clicked list item
        this.classList.add('active');
        
        // Hide all tab panes
        var tabPanes = document.querySelectorAll('.tab-content .tab-pane');
        tabPanes.forEach(function(tabPane) {
          tabPane.classList.remove('show');
          tabPane.classList.remove('active');
        });
        
        // Show the clicked tab pane
        var activeTabPane = document.querySelector(tabPaneId);
        activeTabPane.classList.add('active');
        
        // We need to manually add 'show' class after a slight delay to trigger fade effect
        setTimeout(function() {
          activeTabPane.classList.add('show');
        }, 150); // Adjust delay as needed
      });
    });
  });
  