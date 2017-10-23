export function addPolyfill(polyfill, polyfillLoadedCallback) {
  console.info('Runlevel 0: Polyfill required: ' + polyfill.name);
  const script = document.createElement('script');
  script.src = polyfill.url;
  script.async = false;
  if (polyfillLoadedCallback) {
    script.onload = function() { polyfillLoadedCallback(polyfill); };
  }
  script.onerror = function() {
    new Error('Runlevel 0: Polyfills failed to load script ' + polyfill.name);
  };
  document.head.appendChild(script);
}

export const polyfills = [
  {
    name: 'WebComponents',
    support: function() {
      return 'customElements' in window &&
             'attachShadow' in Element.prototype &&
             'getRootNode' in Element.prototype &&
             'content' in document.createElement('template') &&
             'Promise' in window &&
             'from' in Array;
    },
    url: 'https://distill.pub/third-party/polyfills/webcomponents-lite.js'
  }, {
    name: 'IntersectionObserver',
    support: function() {
      return 'IntersectionObserver' in window &&
             'IntersectionObserverEntry' in window;
    },
    url: 'https://distill.pub/third-party/polyfills/intersection-observer.js'
  },
];

export class Polyfills {

  static browserSupportsAllFeatures() {
    return polyfills.every((poly) => poly.support());
  }

  static load(callback) {
    // Define an intermediate callback that checks if all is loaded.
    const polyfillLoaded = function(polyfill) {
      polyfill.loaded = true;
      console.info('Runlevel 0: Polyfill has finished loading: ' + polyfill.name);
      // console.info(window[polyfill.name]);
      if (Polyfills.neededPolyfills.every((poly) => poly.loaded)) {
        console.info('Runlevel 0: All required polyfills have finished loading.');
        console.info('Runlevel 0->1.');
        window.distillRunlevel = 1;
        callback();
      }
    };
    // Add polyfill script tags
    for (const polyfill of Polyfills.neededPolyfills) {
      addPolyfill(polyfill, polyfillLoaded);
    }
  }

  static get neededPolyfills() {
    if (!Polyfills._neededPolyfills) {
      Polyfills._neededPolyfills = polyfills.filter((poly) => !poly.support());
    }
    return Polyfills._neededPolyfills;
  }
}
