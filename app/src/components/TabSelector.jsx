import React from "react";

const TABS = [
  { id: "no-behavioral", label: "No Behavioural Responses" },
  { id: "behavioral", label: "Behavioural Responses" },
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
