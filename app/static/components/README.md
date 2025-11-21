# Component Library

This directory contains the UI component library for Store Insights, built with React and Tailwind CSS.

## Design System Architecture

This component library uses **semantic design tokens** that support multiple theme modes:

- **Quilt Default** (default theme) - Purple/indigo gradient
- **Liquor** (red/amber theme) - Red/amber gradient  
- **Grocery** (green theme) - Green gradient

Each component references semantic tokens (e.g., `--color-brand-primary`, `--text-heading-md`), and each vertical maps its own colors/typography to those tokens. This gives us:

✅ One shared component library  
✅ Multiple themes without duplicating files  
✅ Easy switching per vertical (similar to shadcn/Tailwind theme logic)  
✅ Updates in the master DS automatically flow to all verticals

## Semantic Tokens

### Color Tokens
- `--color-brand-primary` - Primary brand color
- `--color-brand-primary-foreground` - Primary text color
- `--color-brand-secondary` - Secondary brand color
- `--color-brand-accent` - Accent color
- `--color-background-base` - Base background
- `--color-foreground-base` - Base text color
- `--color-status-destructive` - Error/destructive actions
- `--color-status-success` - Success states
- `--color-status-warning` - Warning states
- `--color-gradient-start` / `--color-gradient-end` - Gradient colors

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
- **component-loader.js** - Loads and processes all component files with Babel

## Usage

### Basic Integration

Components are loaded automatically via the component loader. Here's the required script order:

```html
<!DOCTYPE html>
<html lang="en" data-theme="quilt">
<head>
  <meta charset="UTF-8">
  <title>My App</title>
  
  <!-- Tailwind CSS - Compiled -->
  <link rel="stylesheet" href="/tailwind.css">
</head>
<body>
  <div id="root"></div>
  
  <!-- 1. Load React and Babel -->
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  
  <!-- 2. Load theme switcher and component loader -->
  <script src="/theme-switcher.js"></script>
  <script src="/components/component-loader.js"></script>
  
  <!-- 3. Define your React app (components will be available) -->
  <script type="text/babel">
    const { useState, useEffect } = React;
    
    const MyApp = () => {
      return (
        <div>
          <Button variant="default">Click me</Button>
          <Input placeholder="Enter text" />
          <Badge variant="secondary">New</Badge>
        </div>
      );
    };
    
    // Store render function for delayed execution
    window.renderMyApp = function() {
      if (typeof Button === 'undefined' || typeof Input === 'undefined') {
        setTimeout(window.renderMyApp, 100);
        return;
      }
      ReactDOM.render(<MyApp />, document.getElementById('root'));
    };
  </script>
  
  <!-- 4. Render after components load -->
  <script>
    (function() {
      function tryRender() {
        if (window.componentsLoaded && typeof Button !== 'undefined') {
          if (window.renderMyApp) {
            window.renderMyApp();
          }
        } else {
          setTimeout(tryRender, 50);
        }
      }
      
      // Listen for components loaded event
      window.addEventListener('componentsLoaded', function() {
        if (window.renderMyApp) {
          window.renderMyApp();
        }
      });
      
      // Also try periodically (fallback)
      tryRender();
    })();
  </script>
</body>
</html>
```

### Component Loading

The component loader automatically:
1. Waits for Babel Standalone to be available
2. Fetches all component files (`primitives.js`, `form-components.js`, `display-components.js`, `composite-components.js`)
3. Processes JSX with Babel
4. Executes components in global scope
5. Sets `window.componentsLoaded = true` when complete
6. Dispatches `componentsLoaded` event

## Available Components

### Primitives
- `Button` - Button with variants (default, destructive, outline, secondary, ghost, link) and sizes (sm, default, lg, icon)
- `Input` - Text input field with sizes (sm, default, lg)
- `Label` - Form label
- `Select` - Dropdown select
- `Textarea` - Multi-line text input
- `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent`, `CardFooter` - Card components

### Form Components
- `Checkbox` - Checkbox with label and controlled state
- `RadioGroup`, `RadioGroupItem` - Radio button groups
- `Switch` - Toggle switch with label
- `Slider` - Range slider with min/max/step

### Display Components
- `Badge` - Status badge with variants (default, secondary, destructive, outline)
- `Avatar` - User avatar with fallback and sizes (sm, default, lg, xl)
- `Progress` - Progress bar (0-100)
- `Separator` - Visual separator (horizontal/vertical)
- `Skeleton` - Loading placeholder with pulse animation
- `Alert`, `AlertTitle`, `AlertDescription` - Alert messages with variants

### Composite Components
- `Dialog`, `DialogTrigger`, `DialogContent`, `DialogHeader`, `DialogTitle`, `DialogDescription`, `DialogFooter` - Modal dialogs
- `AlertDialog` and related components - Alert dialogs with actions
- `DropdownMenu` and related components - Dropdown menus
- `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent` - Tab navigation

## Component Examples

### Button

```jsx
<Button variant="default" size="lg">Primary Action</Button>
<Button variant="outline">Secondary</Button>
<Button variant="destructive">Delete</Button>
<Button variant="ghost">Cancel</Button>
```

### Form Elements

```jsx
<Label htmlFor="email">Email</Label>
<Input id="email" type="email" placeholder="you@example.com" />

<Checkbox 
  id="terms" 
  label="Accept terms" 
  checked={accepted}
  onCheckedChange={setAccepted}
/>

<Switch 
  id="notifications"
  label="Enable notifications"
  checked={enabled}
  onCheckedChange={setEnabled}
/>
```

### Cards

```jsx
<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
    <CardDescription>Card description text</CardDescription>
  </CardHeader>
  <CardContent>
    <p>Card content goes here</p>
  </CardContent>
  <CardFooter>
    <Button>Action</Button>
  </CardFooter>
</Card>
```

## Adding New Themes

To add a new theme (e.g., "Pharmacy"):

1. **Add theme colors to `styles.css`:**
```css
[data-theme="pharmacy"] {
  --color-brand-primary: 280 70% 50%;
  --color-brand-primary-foreground: 0 0% 100%;
  --color-brand-secondary: 280 20% 96%;
  --color-brand-secondary-foreground: 280 30% 20%;
  --color-brand-accent: 280 60% 70%;
  --color-brand-accent-foreground: 280 30% 20%;
  --color-gradient-start: 280 70% 50%;
  --color-gradient-end: 280 60% 40%;
}
```

2. **Add dark mode variant:**
```css
[data-theme="pharmacy"].dark,
.dark[data-theme="pharmacy"] {
  --color-background-base: 280 30% 8%;
  --color-foreground-base: 0 0% 98%;
  /* ... other dark mode colors */
}
```

3. **Add to ThemeSwitcher in `theme-switcher.js`:**
```javascript
themes: {
  quilt: 'Quilt Default',
  liquor: 'Liquor',
  grocery: 'Grocery',
  pharmacy: 'Pharmacy', // Add here
}
```

4. **Components automatically adapt!** No component code changes needed.

## Theme-Aware Styling

### Background Gradients

The login page uses theme gradients automatically:

```css
body {
  background: linear-gradient(135deg, 
    hsl(var(--color-gradient-start)) 0%, 
    hsl(var(--color-gradient-end)) 100%
  );
}
```

### Component Colors

All components use semantic tokens, so they automatically adapt:

```jsx
<Button variant="default">  {/* Uses --color-brand-primary */}
<Badge variant="secondary"> {/* Uses --color-brand-secondary */}
<Input />                    {/* Uses --color-border-input */}
```

## Figma Integration

This structure mirrors Figma Variables:
- Semantic tokens in CSS = Figma Variables
- Theme modes (`data-theme`) = Figma Theme Modes
- Component references = Figma Component Properties

When you update tokens in Figma, export the values and update the corresponding CSS variables in `styles.css`. All components will automatically reflect the changes.

## Troubleshooting

### Components Not Loading

1. **Check browser console** for errors
2. **Verify component files exist** at `/components/*.js`
3. **Check network tab** - component files should return 200 status
4. **Verify Babel is loaded** before component-loader.js

### "Button is not defined" Error

This means components haven't loaded yet. The render script should wait automatically, but you can check:

```javascript
console.log('Components loaded:', window.componentsLoaded);
console.log('Button available:', typeof Button);
```

### Theme Not Applying

1. **Check `data-theme` attribute** on `<html>` tag
2. **Verify theme tokens** are defined in `styles.css`
3. **Check localStorage** for saved theme: `localStorage.getItem('store-insights-theme')`

## Files Reference

- `components/primitives.js` - Base components
- `components/form-components.js` - Form elements
- `components/display-components.js` - Display components
- `components/composite-components.js` - Complex components
- `components/component-loader.js` - Component loading system
- `theme-switcher.js` - Theme management utility
- `styles.css` - Design tokens and theme definitions
- `tailwind.css` - Compiled Tailwind CSS (run `npm run build:css:prod`)
