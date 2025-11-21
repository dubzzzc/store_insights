# Component Library

This directory contains the UI component library for Store Insights, built with React and Tailwind CSS.

## Design System Architecture

This component library uses **semantic design tokens** that support multiple theme modes:

- **Quilt Default** (default theme)
- **Liquor** (red/amber theme)
- **Grocery** (green theme)

Each component references semantic tokens (e.g., `color.brand.primary`, `text.heading.md`), and each vertical maps its own colors/typography to those tokens. This gives us:

✅ One shared component library  
✅ Multiple themes without duplicating files  
✅ Easy switching per vertical (similar to shadcn/Tailwind theme logic)  
✅ Updates in the master DS automatically flow to all verticals

## Semantic Tokens

### Color Tokens
- `--color-brand-primary` - Primary brand color
- `--color-brand-secondary` - Secondary brand color
- `--color-brand-accent` - Accent color
- `--color-background-base` - Base background
- `--color-foreground-base` - Base text color
- `--color-status-destructive` - Error/destructive actions
- `--color-status-success` - Success states
- `--color-status-warning` - Warning states

### Typography Tokens
- `--text-heading-xl` through `--text-heading-xs`
- `--text-body-lg` through `--text-body-xs`
- `--text-weight-*` (bold, semibold, medium, normal)
- `--text-line-height-*` (tight, normal, relaxed)

### Spacing Tokens
- `--spacing-1` through `--spacing-16` (8pt grid system)

## Theme Switching

### JavaScript API

```javascript
// Set theme
ThemeSwitcher.setTheme('liquor'); // or 'quilt', 'grocery'

// Toggle dark mode
ThemeSwitcher.toggleDarkMode();

// Get current theme
const currentTheme = ThemeSwitcher.getCurrentTheme();
```

### React Component

```jsx
<ThemeSwitcher.ThemeSelector 
  currentTheme="quilt"
  onThemeChange={(theme) => console.log('Theme changed:', theme)}
  showDarkMode={true}
/>
```

### HTML Attribute

```html
<html data-theme="liquor" class="dark">
  <!-- Components automatically adapt to theme -->
</html>
```

## Structure

- **primitives.js** - Base components (Button, Input, Label, Select, Textarea, Card)
- **form-components.js** - Form elements (Checkbox, Radio, Switch, Slider)
- **display-components.js** - Display elements (Badge, Avatar, Progress, Separator, Alert, Skeleton)
- **composite-components.js** - Complex components (Dialog, AlertDialog, DropdownMenu, Tabs)

## Usage

Components are React components that work with Babel Standalone. Load them in your HTML files:

```html
<!-- Load React and Babel first -->
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

<!-- Load Tailwind CSS -->
<link rel="stylesheet" href="/tailwind.css">

<!-- Load theme switcher -->
<script src="/theme-switcher.js"></script>

<!-- Load component library -->
<script type="text/babel" src="/components/primitives.js"></script>
<script type="text/babel" src="/components/form-components.js"></script>
<script type="text/babel" src="/components/display-components.js"></script>
<script type="text/babel" src="/components/composite-components.js"></script>

<!-- Use components in your React code -->
<script type="text/babel">
  const MyComponent = () => {
    return (
      <div>
        <Button variant="default">Click me</Button>
        <Input placeholder="Enter text" />
        <Badge variant="secondary">New</Badge>
        <ThemeSwitcher.ThemeSelector />
      </div>
    );
  };
</script>
```

## Available Components

### Primitives
- `Button` - Button with variants (default, destructive, outline, secondary, ghost, link)
- `Input` - Text input field
- `Label` - Form label
- `Select` - Dropdown select
- `Textarea` - Multi-line text input
- `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent`, `CardFooter` - Card components

### Form Components
- `Checkbox` - Checkbox with label
- `RadioGroup`, `RadioGroupItem` - Radio button groups
- `Switch` - Toggle switch
- `Slider` - Range slider

### Display Components
- `Badge` - Status badge with variants
- `Avatar` - User avatar with fallback
- `Progress` - Progress bar
- `Separator` - Visual separator
- `Skeleton` - Loading placeholder
- `Alert`, `AlertTitle`, `AlertDescription` - Alert messages

### Composite Components
- `Dialog`, `DialogTrigger`, `DialogContent`, `DialogHeader`, `DialogTitle`, `DialogDescription`, `DialogFooter` - Modal dialogs
- `AlertDialog` and related components - Alert dialogs
- `DropdownMenu` and related components - Dropdown menus
- `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent` - Tab navigation

## Adding New Themes

To add a new theme (e.g., "Pharmacy"):

1. Add theme colors to `styles.css`:
```css
[data-theme="pharmacy"] {
  --color-brand-primary: 280 70% 50%; /* Your brand color */
  --color-brand-primary-foreground: 0 0% 100%;
  /* ... other color mappings */
}
```

2. Add to ThemeSwitcher:
```javascript
themes: {
  quilt: 'Quilt Default',
  liquor: 'Liquor',
  grocery: 'Grocery',
  pharmacy: 'Pharmacy', // Add here
}
```

3. Components automatically adapt! No component code changes needed.

## Figma Integration

This structure mirrors Figma Variables:
- Semantic tokens in CSS = Figma Variables
- Theme modes (`data-theme`) = Figma Theme Modes
- Component references = Figma Component Properties

When you update tokens in Figma, export the values and update the corresponding CSS variables. All components will automatically reflect the changes.
