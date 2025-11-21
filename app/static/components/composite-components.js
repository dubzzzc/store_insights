/**
 * Composite UI Components
 * Complex interactive components built from primitives
 */

// Dialog Component
const Dialog = ({ open = false, onOpenChange, children, ...props }) => {
  const [isOpen, setIsOpen] = React.useState(open);
  
  React.useEffect(() => {
    setIsOpen(open);
  }, [open]);
  
  React.useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);
  
  const handleOpenChange = (newOpen) => {
    setIsOpen(newOpen);
    if (onOpenChange) {
      onOpenChange(newOpen);
    }
  };
  
  if (!isOpen) return null;
  
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      onClick={() => handleOpenChange(false)}
      {...props}
    >
      <div className="fixed inset-0 bg-black/50" />
      <div
        className="relative z-50 w-full max-w-lg rounded-lg border bg-background p-6 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        {React.Children.map(children, (child) => {
          if (React.isValidElement(child)) {
            return React.cloneElement(child, { onOpenChange: handleOpenChange });
          }
          return child;
        })}
      </div>
    </div>
  );
};

const DialogTrigger = ({ children, asChild = false, ...props }) => {
  return React.cloneElement(children, props);
};

const DialogContent = ({ children, className = "", onOpenChange, ...props }) => {
  return (
    <div className={className} {...props}>
      {children}
      <button
        onClick={() => onOpenChange && onOpenChange(false)}
        className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none"
      >
        <span className="sr-only">Close</span>
        <span className="text-2xl">×</span>
      </button>
    </div>
  );
};

const DialogHeader = ({ children, className = "", ...props }) => (
  <div className={`flex flex-col space-y-1.5 text-center sm:text-left ${className}`} {...props}>
    {children}
  </div>
);

const DialogTitle = ({ children, className = "", ...props }) => (
  <h2 className={`text-lg font-semibold leading-none tracking-tight ${className}`} {...props}>
    {children}
  </h2>
);

const DialogDescription = ({ children, className = "", ...props }) => (
  <p className={`text-sm text-muted-foreground ${className}`} {...props}>
    {children}
  </p>
);

const DialogFooter = ({ children, className = "", ...props }) => (
  <div className={`flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2 ${className}`} {...props}>
    {children}
  </div>
);

// Alert Dialog Component
const AlertDialog = ({ open = false, onOpenChange, children, ...props }) => {
  return <Dialog open={open} onOpenChange={onOpenChange} {...props}>{children}</Dialog>;
};

const AlertDialogTrigger = DialogTrigger;
const AlertDialogContent = ({ children, className = "", ...props }) => (
  <DialogContent className={className} {...props}>
    {children}
  </DialogContent>
);
const AlertDialogHeader = DialogHeader;
const AlertDialogTitle = DialogTitle;
const AlertDialogDescription = DialogDescription;
const AlertDialogFooter = DialogFooter;
const AlertDialogAction = ({ children, onClick, ...props }) => (
  <Button onClick={onClick} variant="default" {...props}>
    {children}
  </Button>
);
const AlertDialogCancel = ({ children, onClick, ...props }) => (
  <Button onClick={onClick} variant="outline" {...props}>
    {children}
  </Button>
);

// Dropdown Menu Component
const DropdownMenu = ({ open = false, onOpenChange, children, ...props }) => {
  const [isOpen, setIsOpen] = React.useState(open);
  const menuRef = React.useRef(null);
  
  React.useEffect(() => {
    setIsOpen(open);
  }, [open]);
  
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsOpen(false);
        if (onOpenChange) {
          onOpenChange(false);
        }
      }
    };
    
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onOpenChange]);
  
  if (!isOpen) return null;
  
  return (
    <div
      ref={menuRef}
      className="absolute z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md"
      {...props}
    >
      {children}
    </div>
  );
};

const DropdownMenuTrigger = ({ children, asChild = false, ...props }) => {
  return React.cloneElement(children, props);
};

const DropdownMenuContent = ({ children, className = "", ...props }) => (
  <div className={className} {...props}>
    {children}
  </div>
);

const DropdownMenuItem = ({ children, onClick, className = "", ...props }) => (
  <div
    className={`relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 ${className}`}
    onClick={onClick}
    {...props}
  >
    {children}
  </div>
);

const DropdownMenuLabel = ({ children, className = "", ...props }) => (
  <div className={`px-2 py-1.5 text-sm font-semibold ${className}`} {...props}>
    {children}
  </div>
);

const DropdownMenuSeparator = ({ className = "", ...props }) => (
  <div className={`-mx-1 my-1 h-px bg-muted ${className}`} {...props} />
);

// Tabs Component
const Tabs = ({ defaultValue, value, onValueChange, children, className = "", ...props }) => {
  const [activeTab, setActiveTab] = React.useState(value || defaultValue);
  
  React.useEffect(() => {
    if (value !== undefined) {
      setActiveTab(value);
    }
  }, [value]);
  
  const handleValueChange = (newValue) => {
    setActiveTab(newValue);
    if (onValueChange) {
      onValueChange(newValue);
    }
  };
  
  return (
    <TabsContext.Provider value={{ value: activeTab, onValueChange: handleValueChange }}>
      <div className={className} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
};

const TabsList = ({ children, className = "", ...props }) => (
  <div
    className={`inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground ${className}`}
    {...props}
  >
    {children}
  </div>
);

// Context for Tabs (must be defined before use)
const TabsContext = React.createContext(null);

const TabsTrigger = ({ value, children, className = "", ...props }) => {
  const parent = React.useContext(TabsContext);
  const isActive = parent?.value === value;
  
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${
        isActive ? 'bg-background text-foreground shadow-sm' : ''
      } ${className}`}
      onClick={() => parent?.onValueChange && parent.onValueChange(value)}
      {...props}
    >
      {children}
    </button>
  );
};

const TabsContent = ({ value, children, className = "", ...props }) => {
  const parent = React.useContext(TabsContext);
  if (parent?.value !== value) return null;
  
  return (
    <div
      className={`mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

// Export composite components
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Dialog,
    DialogTrigger,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogFooter,
    AlertDialog,
    AlertDialogTrigger,
    AlertDialogContent,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogAction,
    AlertDialogCancel,
    DropdownMenu,
    DropdownMenuTrigger,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    Tabs,
    TabsList,
    TabsTrigger,
    TabsContent,
    TabsContext,
  };
}

