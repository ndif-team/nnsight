require.config({
    paths: {
        'three': 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js?MEH=BLAH',
        'VANTA': 'https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.net.min.js?MEH=BLAH'
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
        if (VANTA.NET) {
            VANTA.NET({
                el: "#front-background",
                mouseControls: false,
                touchControls: false,
                gyroControls: false,
                minHeight: 200.00,
                minWidth: 200.00,
                scale: 1.00,
                scaleMobile: 1.00,
                maxDistance: 25.00,
                spacing: 20.00,
                points: 5.00,
                backgroundColor: 0xffffff
            })
        }
    });
});
