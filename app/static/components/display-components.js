/**
 * Display UI Components
 * Visual elements for showing information
 */

// Badge Component
const Badge = ({ children, variant = "default", className = "", ...props }) => {
  const variants = {
    default: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
    secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
    destructive: "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
    outline: "text-foreground",
  };
  
  return (
    <div
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

// Avatar Component
const Avatar = ({ src, alt, fallback, className = "", size = "default", ...props }) => {
  const [imgError, setImgError] = React.useState(false);
  
  const sizeClasses = {
    default: "h-10 w-10",
    sm: "h-8 w-8",
    lg: "h-12 w-12",
    xl: "h-16 w-16",
  };
  
  return (
    <div className={`relative flex ${sizeClasses[size]} shrink-0 overflow-hidden rounded-full ${className}`} {...props}>
      {src && !imgError ? (
        <img
          src={src}
          alt={alt}
          onError={() => setImgError(true)}
          className="aspect-square h-full w-full"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center rounded-full bg-muted">
          <span className="text-sm font-medium text-muted-foreground">
            {fallback || (alt ? alt.charAt(0).toUpperCase() : '?')}
          </span>
        </div>
      )}
    </div>
  );
};

// Progress Component
const Progress = ({ value = 0, max = 100, className = "", ...props }) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  
  return (
    <div
      className={`relative h-4 w-full overflow-hidden rounded-full bg-secondary ${className}`}
      {...props}
    >
      <div
        className="h-full w-full flex-1 bg-primary transition-all"
        style={{ transform: `translateX(-${100 - percentage}%)` }}
      />
    </div>
  );
};

// Separator Component
const Separator = ({ orientation = "horizontal", className = "", ...props }) => {
  const orientationClasses = orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]";
  
  return (
    <div
      className={`shrink-0 bg-border ${orientationClasses} ${className}`}
      role="separator"
      aria-orientation={orientation}
      {...props}
    />
  );
};

// Skeleton Component (Loading placeholder)
const Skeleton = ({ className = "", ...props }) => (
  <div
    className={`animate-pulse rounded-md bg-muted ${className}`}
    {...props}
  />
);

// Alert Component
const Alert = ({ children, variant = "default", className = "", ...props }) => {
  const variants = {
    default: "bg-background text-foreground",
    destructive: "border-destructive/50 text-destructive dark:border-destructive [&>svg]:text-destructive",
  };
  
  return (
    <div
      className={`relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground ${variants[variant]} ${className}`}
      role="alert"
      {...props}
    >
      {children}
    </div>
  );
};

const AlertTitle = ({ children, className = "", ...props }) => (
  <h5 className={`mb-1 font-medium leading-none tracking-tight ${className}`} {...props}>
    {children}
  </h5>
);

const AlertDescription = ({ children, className = "", ...props }) => (
  <div className={`text-sm [&_p]:leading-relaxed ${className}`} {...props}>
    {children}
  </div>
);

// Export display components
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Badge,
    Avatar,
    Progress,
    Separator,
    Skeleton,
    Alert,
    AlertTitle,
    AlertDescription,
  };
}

