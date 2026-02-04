import React from "react";

const TABS = [
  { id: "no-behavioral", label: "No Behavioral Responses" },
  { id: "behavioral", label: "Behavioral Responses" },
];

export default function TabSelector({ activeTab, onChange }) {
  return (
    <div className="tab-selector">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
