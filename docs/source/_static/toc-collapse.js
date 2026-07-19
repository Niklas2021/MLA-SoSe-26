// Macht das Inhaltsverzeichnis auf der Projektseite einklappbar.
// docutils rendert .. contents:: als verschachtelte <ul>; hier bekommt jeder
// Eintrag mit Unterpunkten ein Dreieck davor.

document.addEventListener("DOMContentLoaded", function () {
  var toc = document.querySelector(".word-toc");
  if (!toc) return;

  // schaltet die Kapitelnummerierung in custom.css frei
  document.body.classList.add("has-word-toc");

  var mitKindern = [];
  toc.querySelectorAll("li").forEach(function (li) {
    var sub = li.querySelector(":scope > ul");
    if (!sub) return;
    mitKindern.push(li);

    var t = document.createElement("span");
    t.className = "toc-toggle";
    t.textContent = "▾";
    t.setAttribute("role", "button");
    t.setAttribute("tabindex", "0");
    t.setAttribute("aria-expanded", "true");

    function um() {
      var zu = li.classList.toggle("collapsed");
      t.textContent = zu ? "▸" : "▾";
      t.setAttribute("aria-expanded", String(!zu));
    }

    t.addEventListener("click", um);
    t.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        um();
      }
    });
    li.insertBefore(t, li.firstChild);
  });

  if (!mitKindern.length) return;

  // Alles auf/zu -- bei 22 Kapiteln will man nicht einzeln klicken
  var leiste = document.createElement("div");
  leiste.className = "toc-controls";

  function knopf(text, zu) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", function () {
      mitKindern.forEach(function (li) {
        li.classList.toggle("collapsed", zu);
        var t = li.querySelector(":scope > .toc-toggle");
        if (t) {
          t.textContent = zu ? "▸" : "▾";
          t.setAttribute("aria-expanded", String(!zu));
        }
      });
    });
    return b;
  }

  leiste.appendChild(knopf("alle aufklappen", false));
  leiste.appendChild(knopf("alle zuklappen", true));

  var titel = toc.querySelector("p.topic-title");
  if (titel) {
    titel.parentNode.insertBefore(leiste, titel.nextSibling);
  } else {
    toc.insertBefore(leiste, toc.firstChild);
  }

  // Start: alles zu, nur die Hauptteile stehen da
  mitKindern.forEach(function (li) {
    li.classList.add("collapsed");
    var t = li.querySelector(":scope > .toc-toggle");
    if (t) {
      t.textContent = "▸";
      t.setAttribute("aria-expanded", "false");
    }
  });
});
