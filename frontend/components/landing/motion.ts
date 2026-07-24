export const easeOutExpo = [0.22, 1, 0.36, 1] as const;

export const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export const viewportOnce = {
  once: true,
  margin: "-80px",
} as const;
