// Author: Patrick Rowe

export const hourHand = document.getElementById('hour-hand');
export const minuteHand = document.getElementById('minute-hand');
export const secondHand = document.getElementById('second-hand');

function getTime() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const seconds = now.getSeconds();

    return [hours, minutes, seconds]
}

const hoursToDegrees = (hours, minutes) => {
    return hours * 30 + minutes * 0.5
}

const minutesOrSecondsToDegrees = (minutesOrSeconds) => {
    return minutesOrSeconds * 6
}

function convertTimeToAngle(hours, minutes, seconds) {
    const hoursDegrees = hoursToDegrees(hours, minutes);
    const minutesDegrees = minutesOrSecondsToDegrees(minutes);
    const secondsDegrees = minutesOrSecondsToDegrees(seconds);

    return [hoursDegrees, minutesDegrees, secondsDegrees]
}

function rotateHands() {
    const [hours, minutes, seconds] = getTime();
    const [hoursDegrees, minutesDegrees, secondsDegrees] = convertTimeToAngle(hours, minutes, seconds);
    hourHand.style.transform = `rotate(${hoursDegrees}deg)`;
    minuteHand.style.transform = `rotate(${minutesDegrees}deg)`;
    secondHand.style.transform = `rotate(${secondsDegrees}deg)`;
    }

function rotateAllHands() {
    const [hours, minutes, seconds] = getTime();
    const [hoursDegrees, minutesDegrees, secondsDegrees] = convertTimeToAngle(hours, minutes, seconds);

    // Rotate all elements with the "hour hand" class
    const hourHands = document.querySelectorAll('.hour.hand');
    hourHands.forEach(hand => {
        hand.style.transform = `rotate(${hoursDegrees}deg)`;
    });

    // Rotate all elements with the "minute hand" class
    const minuteHands = document.querySelectorAll('.minute.hand');
    minuteHands.forEach(hand => {
        hand.style.transform = `rotate(${minutesDegrees}deg)`;
    });

    // Rotate all elements with the "second hand" class (if needed)
    const secondHands = document.querySelectorAll('.second.hand');
    secondHands.forEach(hand => {
        hand.style.transform = `rotate(${secondsDegrees}deg)`;
    });
}

setInterval(rotateHands, 500);
setInterval(rotateAllHands, 500);