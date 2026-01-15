import { useEffect, useMemo, useState } from 'react';
import Particles, { initParticlesEngine } from '@tsparticles/react';
import { loadSlim } from '@tsparticles/slim';
import type { ISourceOptions } from '@tsparticles/engine';

const ParticleBackground = () => {
  const [inited, setInited] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    }).then(() => setInited(true));
  }, []);

  const options = useMemo(
    () =>
      ({
        background: { color: { value: 'transparent' } },
        fpsLimit: 60,
        interactivity: {
          events: {
            onClick: { enable: true, mode: 'push' },
            onHover: { enable: true, mode: 'repulse' },
            resize: { enable: true, delay: 0 },
          },
          modes: {
            push: { quantity: 2 },
            repulse: { distance: 100, duration: 0.4 },
          },
        },
        particles: {
          color: { value: ['#0070f3', '#00c8ff', '#3291ff'] },
          links: {
            color: '#0070f3',
            distance: 150,
            enable: true,
            opacity: 0.3,
            width: 1,
          },
          move: {
            direction: 'none',
            enable: true,
            outModes: { default: 'bounce' },
            random: false,
            speed: 1,
            straight: false,
          },
          number: {
            density: { enable: true, width: 1920, height: 1080 },
            value: 80,
          },
          opacity: {
            value: { min: 0.1, max: 0.6 },
            animation: {
              enable: true,
              speed: 1,
              sync: false,
              startValue: 'random',
            },
          },
          shape: { type: 'circle' },
          size: {
            value: { min: 1, max: 3 },
            animation: {
              enable: true,
              speed: 2,
              sync: false,
              startValue: 'random',
            },
          },
        },
        detectRetina: true,
      } satisfies ISourceOptions),
    []
  );

  if (!inited) return null;

  return (
    <Particles id="tsparticles" options={options} className="absolute inset-0 -z-10" />
  );
};

export default ParticleBackground;
