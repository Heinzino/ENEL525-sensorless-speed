const pptxgen = require("pptxgenjs");
const path = require("path");

const COLOR = {
  darkBg: "1B2420",
  cream: "F1EDE3",
  beige: "E8DFC8",
  ink: "1B2420",
  body: "3B3F3A",
  muted: "7A7F76",
  mutedLight: "A7A79A",
  forest: "3B5D3A",
  terracotta: "C26A33",
  hairline: "C9C4B4",
  titleOnDark: "EFE9D9",
  subtitleOnDark: "C9C4B4",
  card: "F7F3EA",
};

const FONT = { head: "Georgia", body: "Calibri", mono: "Consolas" };
const FIG = path.join(__dirname, "..", "results", "figures");

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pres.layout = "WIDE";
pres.title = "Sensorless Speed Estimation for a Brushed DC Motor";
pres.author = "Mohammad Elmougey and Heinz Veintimilla";
pres.subject = "ENEL 525 course project";

const W = 13.333;
const H = 7.5;
const M = 0.7;

function eyebrow(slide, label, page, onDark) {
  const color = onDark ? COLOR.mutedLight : COLOR.muted;
  slide.addText(label, {
    x: M, y: 0.32, w: 8.5, h: 0.28,
    fontFace: FONT.body, fontSize: 12, color, margin: 0, charSpacing: 1.5,
  });
  slide.addText(page, {
    x: W - M - 1.6, y: 0.32, w: 1.6, h: 0.28,
    fontFace: FONT.body, fontSize: 12, color, align: "right", margin: 0,
  });
}

function title(slide, text, y) {
  slide.addText(text, {
    x: M, y: y || 0.68, w: W - 2 * M, h: 0.62,
    fontFace: FONT.head, fontSize: 32, color: COLOR.ink, margin: 0,
  });
}

function notes(slide, text) {
  slide.addNotes(text);
}

function slideTitle() {
  const s = pres.addSlide();
  s.background = { color: COLOR.darkBg };
  eyebrow(s, "ENEL 525  ·  MACHINE LEARNING FOR ENGINEERS", "01", true);
  s.addText("Sensorless speed estimation\nfor a brushed DC motor", {
    x: M, y: 1.9, w: 11.2, h: 2.2,
    fontFace: FONT.head, fontSize: 44, color: COLOR.titleOnDark, margin: 0,
  });
  s.addText("Polynomial regression and a one-hidden-layer network,\nscored against the state-space speed.", {
    x: M, y: 4.4, w: 10, h: 0.9,
    fontFace: FONT.body, fontSize: 18, color: COLOR.subtitleOnDark, margin: 0,
  });
  s.addText("Mohammad Elmougey    Heinz Veintimilla\nUniversity of Calgary", {
    x: M, y: 6.15, w: 8, h: 0.7,
    fontFace: FONT.body, fontSize: 16, color: COLOR.titleOnDark, margin: 0,
  });
  notes(s, "About 45 seconds. Name the course, the motor, and the claim: we try to beat the state-space speed without an encoder.");
}

function slideProblem() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "PROBLEM", "02");
  title(s, "The wheelchair needs speed, not an encoder");
  const cards = [
    ["01", "The rider", "Head tilts command a chair for someone who cannot correct the loop by hand."],
    ["02", "The sensor", "A shaft encoder costs space and money, and shock, heat, and moisture move its reading."],
    ["03", "The state", "Current is measured on the drive. Shaft speed is the state the controller still needs."],
  ];
  cards.forEach((card, i) => {
    const x = M + i * 4.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.8, w: 3.85, h: 4.5,
      fill: { color: COLOR.card },
      line: { color: COLOR.hairline, width: 1 },
    });
    s.addText(card[0], {
      x: x + 0.28, y: 2.05, w: 3.2, h: 0.4,
      fontFace: FONT.mono, fontSize: 14, color: COLOR.terracotta, margin: 0,
    });
    s.addText(card[1], {
      x: x + 0.28, y: 2.6, w: 3.2, h: 0.9,
      fontFace: FONT.head, fontSize: 26, color: COLOR.ink, margin: 0,
    });
    s.addText(card[2], {
      x: x + 0.28, y: 3.7, w: 3.2, h: 2.0,
      fontFace: FONT.body, fontSize: 16, color: COLOR.body, margin: 0,
    });
  });
  notes(s, "One minute. The ethical point is the rider, not a generic safety slide. Speed is the missing state.");
}

function slideTask() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "LEARNING TASK", "03");
  title(s, "Predict measured speed at one instant");
  s.addText("Inputs, standardized on the training slice only", {
    x: M, y: 1.6, w: 7, h: 0.35,
    fontFace: FONT.body, fontSize: 16, color: COLOR.muted, margin: 0,
  });
  const inputs = [
    ["ia", "Measured armature current"],
    ["i model", "State-space current"],
    ["ω model", "State-space speed, also the baseline"],
  ];
  inputs.forEach((row, i) => {
    const y = 2.15 + i * 1.15;
    s.addShape(pres.shapes.RECTANGLE, {
      x: M, y, w: 7.3, h: 1.0,
      fill: { color: COLOR.card },
      line: { color: COLOR.hairline, width: 1 },
    });
    s.addText(row[0], {
      x: M + 0.25, y, w: 1.8, h: 1.0,
      fontFace: FONT.mono, fontSize: 18, color: COLOR.forest, valign: "middle", margin: 0,
    });
    s.addText(row[1], {
      x: M + 2.2, y, w: 4.8, h: 1.0,
      fontFace: FONT.body, fontSize: 18, color: COLOR.ink, valign: "middle", margin: 0,
    });
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.5, y: 2.15, w: 4.1, h: 3.45,
    fill: { color: COLOR.darkBg },
  });
  s.addText("TARGET", {
    x: 8.75, y: 2.4, w: 3.6, h: 0.3,
    fontFace: FONT.body, fontSize: 13, color: COLOR.mutedLight, margin: 0,
  });
  s.addText("ω measured", {
    x: 8.75, y: 2.9, w: 3.6, h: 0.7,
    fontFace: FONT.head, fontSize: 28, color: COLOR.titleOnDark, margin: 0,
  });
  s.addText("Radians per second.\nNo lag window.\nNo reconstructed voltage.", {
    x: 8.75, y: 3.8, w: 3.6, h: 1.4,
    fontFace: FONT.body, fontSize: 16, color: COLOR.subtitleOnDark, margin: 0,
  });
  notes(s, "One minute. Stress instantaneous regression and that the baseline is one of the inputs.");
}

function slideData() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "DATA", "04");
  title(s, "Two 40-second runs, one locked");
  const facts = [
    ["40,001", "rows after alignment to 1 ms"],
    ["32,000", "training rows, t < 32 s"],
    ["8,001", "validation rows, same recording"],
    ["40,001", "test rows, untouched until selection"],
  ];
  facts.forEach((fact, i) => {
    const y = 1.7 + i * 1.2;
    s.addText(fact[0], {
      x: M, y, w: 2.6, h: 0.7,
      fontFace: FONT.head, fontSize: 32, color: COLOR.terracotta, margin: 0,
    });
    s.addText(fact[1], {
      x: 3.4, y, w: 5.2, h: 0.7,
      fontFace: FONT.body, fontSize: 20, color: COLOR.ink, valign: "middle", margin: 0,
    });
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.7, y: 1.7, w: 3.9, h: 4.6,
    fill: { color: COLOR.beige },
  });
  s.addText("Held fixed", {
    x: 8.95, y: 1.95, w: 3.4, h: 0.4,
    fontFace: FONT.head, fontSize: 22, color: COLOR.ink, margin: 0,
  });
  s.addText("Measured traces arrive at 50 µs and are sampled onto the model grid. Nothing is interpolated.\n\nThe test run is a smaller motion, about ±6 rad/s against about −13 to +11 on the development run.\n\nCurrent is treated as amperes.", {
    x: 8.95, y: 2.55, w: 3.4, h: 3.4,
    fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
  });
  notes(s, "One minute. Say the cut is chronological and the test files were not opened for tuning.");
}

function slideEda() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "DEVELOPMENT RECORDING", "05");
  title(s, "The model follows the shape and misses the plateaus");
  s.addImage({
    path: path.join(FIG, "eda_development.png"),
    x: 0.55, y: 1.55, w: 7.62, h: 5.5,
  });
  s.addText("The dashed line is the training cut. Validation is the last plateau, so a low validation error is not the whole story.", {
    x: 9.2, y: 2.2, w: 3.4, h: 3.2,
    fontFace: FONT.body, fontSize: 18, color: COLOR.body, margin: 0,
  });
  notes(s, "45 seconds. Point at the offset between black and blue on the current plot, and at 32 seconds.");
}

function slideMethods() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "METHODS", "06");
  title(s, "Baseline, polynomials, and one network");
  const cols = [
    [COLOR.forest, "Baseline", "Use the state-space speed as the prediction. No fitted weights."],
    [COLOR.terracotta, "Polynomials", "Full degree 2–5, with cross terms. Least squares, and plain SGD at three learning rates."],
    [COLOR.ink, "Network", "16 or 32 ReLU units, linear output, same three inputs. Written in NumPy."],
  ];
  cols.forEach((col, i) => {
    const x = M + i * 4.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.85, w: 0.12, h: 4.3,
      fill: { color: col[0] },
    });
    s.addText(col[1], {
      x: x + 0.4, y: 1.9, w: 3.4, h: 0.7,
      fontFace: FONT.head, fontSize: 26, color: COLOR.ink, margin: 0,
    });
    s.addText(col[2], {
      x: x + 0.4, y: 2.8, w: 3.4, h: 2.4,
      fontFace: FONT.body, fontSize: 18, color: COLOR.body, margin: 0,
    });
  });
  s.addText("100 epochs  ·  batch 256  ·  selection seed 525  ·  winner = lowest validation RMSE", {
    x: M, y: 6.55, w: 12, h: 0.35,
    fontFace: FONT.body, fontSize: 16, color: COLOR.muted, margin: 0,
  });
  notes(s, "One minute. Say least squares and SGD are two trainers for one polynomial, and the network is the second model class.");
}

function slideVal() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "VALIDATION", "07");
  title(s, "Degree 3 and a small step size win");
  s.addTable([
    [
      { text: "Configuration", options: { fill: { color: COLOR.darkBg }, color: COLOR.titleOnDark, bold: true } },
      { text: "Val. RMSE", options: { fill: { color: COLOR.darkBg }, color: COLOR.titleOnDark, bold: true, align: "right" } },
    ],
    ["State-space speed", { text: "1.113", options: { align: "right" } }],
    ["Least squares, degree 2", { text: "1.452", options: { align: "right" } }],
    ["Least squares, degree 5", { text: "137.7", options: { align: "right" } }],
    [
      { text: "SGD, degree 3, lr 0.001", options: { fill: { color: "E7F0E4" }, bold: true } },
      { text: "0.387", options: { fill: { color: "E7F0E4" }, align: "right", bold: true } },
    ],
    ["Network, 32 units, lr 0.01", { text: "0.713", options: { align: "right" } }],
  ], {
    x: M, y: 1.7, w: 7.4, h: 4.6,
    colW: [5.0, 2.4],
    fontFace: FONT.body,
    fontSize: 16,
    color: COLOR.ink,
    border: [{ pt: 0 }, { pt: 0 }, { pt: 0 }, { pt: 0 }],
    valign: "middle",
  });
  s.addText("Higher degree lowers training error and destroys the validation error. Learning rate 0.1 diverges.", {
    x: 8.5, y: 2.1, w: 4.1, h: 2.2,
    fontFace: FONT.body, fontSize: 18, color: COLOR.body, margin: 0,
  });
  s.addText("Condition number at degree 5: 3.9×10¹⁵.", {
    x: 8.5, y: 4.5, w: 4.1, h: 1.2,
    fontFace: FONT.head, fontSize: 22, color: COLOR.terracotta, margin: 0,
  });
  notes(s, "One minute. The bold row is the selection. Degree 5 is the trap if someone picks by training error.");
}

function slideCurves() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "LEARNING CURVES", "08");
  title(s, "The epoch budget is part of the method");
  s.addImage({
    path: path.join(FIG, "learning_curves.png"),
    x: 0.55, y: 1.85, w: 8.2, h: 3.64,
  });
  s.addText("SGD validation error returns to its best region only near epoch 100.\n\nThe network is best near epoch 20, then the training error keeps falling. We still keep epoch 100.", {
    x: 9.15, y: 1.9, w: 3.5, h: 4.2,
    fontFace: FONT.body, fontSize: 16, color: COLOR.body, margin: 0,
  });
  notes(s, "One minute. Point at the network's early minimum and say we did not early-stop.");
}

function slideTest() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "LOCKED TEST", "09");
  title(s, "Only the cubic SGD model beats the baseline");
  const stats = [
    ["0.890", "State-space baseline"],
    ["1.061", "Best least squares"],
    ["0.768", "SGD, mean ± 0.043"],
    ["1.210", "Network, mean ± 0.154"],
  ];
  stats.forEach((stat, i) => {
    const x = M + (i % 2) * 6.1;
    const y = 1.8 + Math.floor(i / 2) * 2.3;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 5.8, h: 2.05,
      fill: { color: i === 2 ? COLOR.darkBg : COLOR.card },
    });
    s.addText(stat[0], {
      x: x + 0.3, y: y + 0.25, w: 5.2, h: 0.85,
      fontFace: FONT.head, fontSize: 40,
      color: i === 2 ? COLOR.titleOnDark : COLOR.ink, margin: 0,
    });
    s.addText(stat[1], {
      x: x + 0.3, y: y + 1.15, w: 5.2, h: 0.5,
      fontFace: FONT.body, fontSize: 18,
      color: i === 2 ? COLOR.subtitleOnDark : COLOR.body, margin: 0,
    });
  });
  notes(s, "One minute. Read the four numbers. The dark card is the result to remember. Units are rad/s.");
}

function slideTrace() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "WHERE THE ERROR GOES", "10");
  title(s, "The average win hides the first 10 seconds");
  s.addImage({
    path: path.join(FIG, "test_trace.png"),
    x: 0.4, y: 1.55, w: 8.0, h: 5.11,
  });
  const blocks = [
    ["0–10 s", "worse", "0.98 vs 0.64"],
    ["20–30 s", "better", "0.59 vs 1.17"],
    ["30–40 s", "better", "0.16 vs 0.59"],
  ];
  blocks.forEach((block, i) => {
    const y = 1.7 + i * 1.55;
    s.addText(block[0], {
      x: 8.9, y, w: 3.8, h: 0.35,
      fontFace: FONT.mono, fontSize: 14, color: COLOR.muted, margin: 0,
    });
    s.addText(block[1], {
      x: 8.9, y: y + 0.32, w: 3.8, h: 0.4,
      fontFace: FONT.head, fontSize: 22,
      color: block[1] === "worse" ? COLOR.terracotta : COLOR.forest, margin: 0,
    });
    s.addText(block[2] + " rad/s", {
      x: 8.9, y: y + 0.78, w: 3.8, h: 0.35,
      fontFace: FONT.body, fontSize: 16, color: COLOR.body, margin: 0,
    });
  });
  notes(s, "One minute. The lower panel is 25 to 35 seconds, where the cubic model is closer to the measurement (0.31 versus 1.14 rad/s) and the state-space trace sits low. The miss is still the first 10 seconds. Seed 525.");
}

function slideLimits() {
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "LIMITS", "11");
  title(s, "A simulation RMSE is not a deployment argument");
  const items = [
    ["Simulated traces", "No passenger load, brush wear, or a wheel leaving the ground."],
    ["Structured residual", "About 1 rad/s of bias on a plateau still moves a speed loop."],
    ["Units", "Amperes and rad/s are inferred from the scale of the files."],
    ["The rider", "This person cannot grab the wheels if the loop is wrong."],
  ];
  items.forEach((item, i) => {
    const y = 1.65 + i * 1.3;
    s.addText(item[0], {
      x: M, y, w: 4.2, h: 0.45,
      fontFace: FONT.head, fontSize: 22, color: COLOR.ink, margin: 0,
    });
    s.addText(item[1], {
      x: 5.2, y, w: 7.3, h: 0.7,
      fontFace: FONT.body, fontSize: 18, color: COLOR.body, margin: 0,
    });
  });
  notes(s, "One minute. Do not soften the last point. Hardware is future work, not a claim.");
}

function slideClose() {
  const s = pres.addSlide();
  s.background = { color: COLOR.darkBg };
  eyebrow(s, "CONCLUSION", "12", true);
  s.addText("Cubic SGD beats the\nstate-space speed.", {
    x: M, y: 1.3, w: 11, h: 1.8,
    fontFace: FONT.head, fontSize: 40, color: COLOR.titleOnDark, margin: 0,
  });
  s.addText("0.768 ± 0.043 rad/s    against    0.890 rad/s", {
    x: M, y: 3.4, w: 11, h: 0.6,
    fontFace: FONT.body, fontSize: 24, color: COLOR.terracotta, margin: 0,
  });
  s.addText("Least squares overfits as the degree rises.\nThe network does not survive the locked test.\nThe error that remains is systematic in time.", {
    x: M, y: 4.4, w: 10, h: 1.5,
    fontFace: FONT.body, fontSize: 20, color: COLOR.subtitleOnDark, margin: 0,
  });
  notes(s, "45 seconds, then take questions. Repeat the two numbers and stop.");
}

slideTitle();
slideProblem();
slideTask();
slideData();
slideEda();
slideMethods();
slideVal();
slideCurves();
slideTest();
slideTrace();
slideLimits();
slideClose();

pres.writeFile({ fileName: path.join(__dirname, "ENEL525_Project_Presentation.pptx") })
  .then(() => console.log("wrote pptx"))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
