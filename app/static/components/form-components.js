/**
 * Form UI Components
 * Interactive form elements
 */

// Checkbox Component
const Checkbox = ({ checked = false, onCheckedChange, className = "", id, label, ...props }) => {
  const [isChecked, setIsChecked] = React.useState(checked);
  
  const handleChange = (e) => {
    const newChecked = e.target.checked;
    setIsChecked(newChecked);
    if (onCheckedChange) {
      onCheckedChange(newChecked);
    }
  };
  
  React.useEffect(() => {
    setIsChecked(checked);
  }, [checked]);
  
  return (
    <div className="flex items-center space-x-2">
      <input
        type="checkbox"
        id={id}
        checked={isChecked}
        onChange={handleChange}
        className={`h-4 w-4 rounded border border-primary text-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
        {...props}
      />
      {label && (
        <label htmlFor={id} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          {label}
        </label>
      )}
    </div>
  );
};

// Radio Group Component
const RadioGroup = ({ value, onValueChange, children, className = "", ...props }) => {
  const [selectedValue, setSelectedValue] = React.useState(value);
  
  React.useEffect(() => {
    setSelectedValue(value);
  }, [value]);
  
  const handleChange = (newValue) => {
    setSelectedValue(newValue);
    if (onValueChange) {
      onValueChange(newValue);
    }
  };
  
  return (
    <div className={`grid gap-2 ${className}`} {...props}>
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, {
            checked: child.props.value === selectedValue,
            onCheckedChange: () => handleChange(child.props.value),
          });
        }
        return child;
      })}
    </div>
  );
};

const RadioGroupItem = ({ id, value, checked = false, onCheckedChange, label, className = "", ...props }) => {
  return (
    <div className="flex items-center space-x-2">
      <input
        type="radio"
        id={id}
        value={value}
        checked={checked}
        onChange={() => onCheckedChange && onCheckedChange()}
        className="h-4 w-4 border border-primary text-primary ring-offset-background focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        {...props}
      />
      {label && (
        <label htmlFor={id} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          {label}
        </label>
      )}
    </div>
  );
};

// Switch Component
const Switch = ({ checked = false, onCheckedChange, id, label, className = "", disabled = false, ...props }) => {
  const [isChecked, setIsChecked] = React.useState(checked);
  
  const handleChange = () => {
    if (disabled) return;
    const newChecked = !isChecked;
    setIsChecked(newChecked);
    if (onCheckedChange) {
      onCheckedChange(newChecked);
    }
  };
  
  React.useEffect(() => {
    setIsChecked(checked);
  }, [checked]);
  
  return (
    <div className="flex items-center space-x-2">
      <button
        type="button"
        role="switch"
        aria-checked={isChecked}
        onClick={handleChange}
        disabled={disabled}
        className={`peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 ${
          isChecked ? 'bg-primary' : 'bg-input'
        } ${className}`}
        {...props}
      >
        <span
          className={`pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform ${
            isChecked ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
      {label && (
        <label htmlFor={id} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          {label}
        </label>
      )}
    </div>
  );
};

// Slider Component
const Slider = ({ value = 50, onValueChange, min = 0, max = 100, step = 1, className = "", ...props }) => {
  const [sliderValue, setSliderValue] = React.useState(value);
  
  const handleChange = (e) => {
    const newValue = Number(e.target.value);
    setSliderValue(newValue);
    if (onValueChange) {
      onValueChange(newValue);
    }
  };
  
  React.useEffect(() => {
    setSliderValue(value);
  }, [value]);
  
  const percentage = ((sliderValue - min) / (max - min)) * 100;
  
  return (
    <div className={`relative flex w-full touch-none select-none items-center ${className}`} {...props}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={sliderValue}
        onChange={handleChange}
        className="sr-only"
      />
      <div className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
        <div
          className="absolute h-full bg-primary"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div
        className="absolute h-5 w-5 -translate-x-1/2 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        style={{ left: `${percentage}%` }}
      />
    </div>
  );
};

// Export form components
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Checkbox,
    RadioGroup,
    RadioGroupItem,
    Switch,
    Slider,
  };
}

