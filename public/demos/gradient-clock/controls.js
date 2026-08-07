// Author: Patrick Rowe

// Already declared in clock.js
// const hourHand = document.getElementById("hour-hand");
// const minuteHand = document.getElementById("minute-hand");

import { hourHand, minuteHand } from "./clock.js";

const hourHandColour1Grp = document.getElementById("hour-hand-colour-1");
const hourHandColour2Grp = document.getElementById("hour-hand-colour-2");
const minuteHandcolour1Grp = document.getElementById("minute-hand-colour-1");
const minuteHandcolour2Grp = document.getElementById("minute-hand-colour-2");


/* Colour Definitions */
const colours = [
    "transparent",
    "black",
    "white",
    "#32a8a2",
    "#9432a8",
    "#4ea832",
    "#a83250",
    "#cfa12d",
    "#cf2d7b",
];

const colorBlendModes = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    // "soft-light",
    "difference",
    // "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity"
];

/* Hand Definitions */
let hourHandColour1;
let hourHandColour2;
let hourHandExtent;
let minuteHandColour1;
let minuteHandColour2;
let minuteHandExtent;

/* Defaults */ 
let clockStyleDefaults = {
    hourHandColour1: colours[1],
    hourHandColour2: colours[2],
    minuteHandColour1: colours[0],
    minuteHandColour2: colours[2],
    hourHandExtent: 40,
    minuteHandExtent: 10,
};

const handColourGrps = [
    hourHandColour1Grp, 
    hourHandColour2Grp, 
    minuteHandcolour1Grp, 
    minuteHandcolour2Grp
];

/* Opening and closing of controls */


const controlsButton = document.getElementById("controls-button-container");
const controlsButtonClose = document.getElementById("controls-button-container-close");


function toggleControls() {
    console.log("toggle controls");
    const controls = document.getElementById("controls");
    controls.classList.toggle("hidden");
    controlsButton.classList.toggle("hidden");
}

controlsButtonClose.addEventListener("click", toggleControls);
controlsButton.addEventListener("click", toggleControls);

/* 
const closeControlsButton = document.getElementById("close-controls");
closeControlsButton.addEventListener("click", toggleControls);
*/

/* Reset controls */


const resetControls = () => {
    console.log("reset controls");
    hourHandColour1 = clockStyleDefaults.hourHandColour1;
    hourHandColour2 = clockStyleDefaults.hourHandColour2;
    minuteHandColour1 = clockStyleDefaults.minuteHandColour1;
    minuteHandColour2 = clockStyleDefaults.minuteHandColour2;
    hourHandExtent = clockStyleDefaults.hourHandExtent;
    minuteHandExtent = clockStyleDefaults.minuteHandExtent;


    hourHandExtentSlider.value = hourHandExtent;
    minuteHandExtentSlider.value = minuteHandExtent;

    updateClockGradients();
    updateExtents();
}

/*
const resetControlsButton = document.getElementById("reset-controls");
resetControlsButton.addEventListener("click", resetControls);
*/

/* set up colour buttons */
const setupColourButtons = () => {
    handColourGrps.forEach(hand => {
        colours.forEach(colour => {
            const id = `${hand.id}-${colour}`
            console.log(hand.id);

            hand.innerHTML += `
                <button class="colour-button" id="${id}" onclick="setHandColours('${hand.id}', '${colour}', '${id}')"></button>    
            `
            
            if (colour === "transparent") {
                document.getElementById(id).style.backgroundImage = "url('img/transparent-checkerboard.png')";
                document.getElementById(id).style.backgroundSize = "200%";
                document.getElementById(id).style.backgroundRepeat = "no-repeat";
                document.getElementById(id).style.backgroundPosition = "center";
            } else {
                document.getElementById(id).style.backgroundColor = colour;
            }
        })
    });
}


/* change colours by clicking on buttons */
function setHandColours (hand, colour, btnId) {
    switch (hand) {
        case "hour-hand-colour-1":
            hourHandColour1 = colour;
            break;
        case "hour-hand-colour-2":
            hourHandColour2 = colour;
            break;
        case "minute-hand-colour-1":
            minuteHandColour1 = colour;
            break;
        case "minute-hand-colour-2":
            minuteHandColour2 = colour;
            break;
    }

    // Remove 'btn-selected' class from hand colour buttons 
    const handColourButtons = document.querySelectorAll(`#${hand} .colour-button`);
    handColourButtons.forEach(button => {
        button.classList.remove("btn-selected");
    }
    )

    const btn = document.getElementById(btnId);
    btn.classList.add("btn-selected");


    updateClockGradients();
}

function updateAllHandColours() {
    // Update all elements with the "hour hand" class
    const hourHands = document.querySelectorAll('.hour.hand');
    hourHands.forEach(hand => {
        hand.style.backgroundImage = `conic-gradient(
            ${hourHandColour1} ${hourHandExtent}%,
            ${hourHandColour2}
        )`;
    });

    // Update all elements with the "minute hand" class
    const minuteHands = document.querySelectorAll('.minute.hand');
    minuteHands.forEach(hand => {
        hand.style.backgroundImage = `conic-gradient(
            ${minuteHandColour1} ${minuteHandExtent}%,
            ${minuteHandColour2}
        )`;
    });
}

/* update clock gradients when colours change */
function updateClockGradients () {

    hourHand.style.backgroundImage = `conic-gradient(
        ${hourHandColour1} ${hourHandExtent}%,
        ${hourHandColour2}
    )
    `
    minuteHand.style.backgroundImage = `conic-gradient(
        ${minuteHandColour1} ${minuteHandExtent}%,
        ${minuteHandColour2}
    )
    `

    console.log("Minuehandextendt", minuteHandExtent);
    console.log("Hourhandextendt", hourHandExtent);

    updateAllHandColours();
}

/* Extents */
const minuteHandExtentSlider = document.getElementById("minute-extent-slider");   
const hourHandExtentSlider = document.getElementById("hour-extent-slider");

minuteHandExtentSlider.addEventListener("input", updateExtents);
hourHandExtentSlider.addEventListener("input", updateExtents);

function updateExtents () {
    hourHandExtent = hourHandExtentSlider.value;
    minuteHandExtent = minuteHandExtentSlider.value;

    updateClockGradients();
}

/* Color Blend Mode */ 
const blendModeButtonGrp = document.getElementById("color-blend-mode");

const setupColourBlendButtons = () => {
    colorBlendModes.forEach(blendMode => {
        const id = `colour-blend-mode-${blendMode}`
        console.log(blendMode);
        blendModeButtonGrp.innerHTML += `
            <button id="${id}" class="blend-button controls-button" onclick="updateBlendMode('${blendMode}')">
                <div id="${blendMode}-clock-face" class="blend-button controls-button clock-face"</div>
                    <div id="${blendMode}-hour-hand" class="blend-button hour hand"></div>
                    <div id="${blendMode}-minute-hand" class="blend-button minute hand"></div>
                    <div id="${blendMode}-cap" class="cap"></div>
                </div> 
                <div id="${blendMode}-tooltip", class="tooltip">
                        ${blendMode}
                </div>
            </button>
            `
                ;

        if (blendMode === "normal") {
            document.getElementById(`${blendMode}-tooltip`).innerText = "No Blend";
        }

        let buttonClockFace = document.getElementById(`${blendMode}-clock-face`);
        let buttonMinuteHand = document.getElementById(`${blendMode}-minute-hand`);
        let buttonHourHand = document.getElementById(`${blendMode}-hour-hand`);
    
        buttonMinuteHand.style.mixBlendMode = blendMode;
        // buttonHourHand.style.mixBlendMode = blendMode;
        buttonClockFace.style.backgroundColor = "white";
        
    });


}

function updateBlendMode (blendMode) {
    minuteHand.style.mixBlendMode = blendMode;

    // add the seleceted class to the selected button
    const blendModeButtons = document.querySelectorAll(`.blend-button`);
    blendModeButtons.forEach(button => {
        button.classList.remove("btn-selected");
    });


    const btn = document.getElementById(`colour-blend-mode-${blendMode}`);
    btn.classList.add("btn-selected");
}

/* Functions needed to be accessible globally */
window.setHandColours = setHandColours;
window.updateBlendMode = updateBlendMode;

/* Initialisation */
setupColourButtons();
setupColourBlendButtons();
resetControls();
updateClockGradients();
updateExtents();
updateBlendMode("hard-light");

