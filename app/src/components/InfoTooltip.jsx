import React, { useState } from "react";

export default function InfoTooltip({ title, description }) {
  const [visible, setVisible] = useState(false);

  return (
    <span
      className="info-tooltip-wrapper"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <span className="info-tooltip-icon">i</span>
      {visible && (
        <div className="info-tooltip-box">
          <div className="info-tooltip-title">{title}</div>
          <div className="info-tooltip-desc">{description}</div>
          <div className="info-tooltip-arrow" />
        </div>
      )}
    </span>
  );
}
